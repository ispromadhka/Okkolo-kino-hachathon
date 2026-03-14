"""
V12 pipeline: answer augmentation + HyDE query expansion.
Score: 0.500 (1st place)

Pipeline:
1. Build 90s/30s window chunks from faster-whisper large-v3-turbo transcripts
2. Add answer_en from train data as augmentation chunks (same timestamps)
3. Embed all with BGE-M3 → numpy index
4. For each test query:
   a. Find most similar train question
   b. If similar enough (>0.7), mix query embedding with train answer embedding (HyDE)
   c. Search with mixed embedding
5. Generate submission with ±10s padding

Usage: python run_v12.py
"""
import logging
import pickle
import csv
import re
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def main():
    t0 = time.time()

    # --- Load data ---
    with open('new_transcripts.pkl', 'rb') as f:
        transcripts = pickle.load(f)

    video_map = {}
    with open('data/video_files.csv') as f:
        for row in csv.DictReader(f):
            m = re.search(r'(video_[a-f0-9]+)', row['video_path'])
            if m:
                video_map[m.group(1)] = row['video_path']
    log.info(f'Loaded {len(transcripts)} videos, {len(video_map)} video paths')

    # --- Build chunks ---
    from pipeline.chunker import merge_segments_to_window
    from pipeline.indexer import get_embed_model, build_index, load_index, search_index

    all_chunks = []

    # 1. Standard 90s/30s window chunks
    for key, segs in transcripts.items():
        if not segs:
            continue
        m = re.search(r'(video_[a-f0-9]+)', key)
        vid = m.group(1) if m else key
        vfile = video_map.get(vid, f'videos/{vid}.mp4')
        windows = merge_segments_to_window(segs, 90.0, 30.0)
        for i, w in enumerate(windows):
            all_chunks.append({
                'video_file': vfile,
                'start_time': w['start'],
                'end_time': w['end'],
                'text': w['text'],
                'chunk_index': i,
                'chunk_type': 'window',
            })
    log.info(f'Window chunks: {len(all_chunks)}')

    # 2. Answer augmentation from train data
    import pandas as pd
    train = pd.read_csv('data/train_qa.csv')
    aug_count = 0
    for _, row in train.iterrows():
        answer = str(row.get('answer_en', '')).strip()
        if not answer or answer == 'nan' or len(answer) < 20:
            continue
        if len(answer) > 1000:
            answer = answer[:1000]
        all_chunks.append({
            'video_file': row['video_file'],
            'start_time': float(row['start']),
            'end_time': float(row['end']),
            'text': answer,
            'chunk_index': 90000 + aug_count,
            'chunk_type': 'answer_aug',
        })
        aug_count += 1
    log.info(f'Answer augmentation: +{aug_count} chunks')
    log.info(f'Total chunks: {len(all_chunks)}')

    # 3. Embed and save index
    build_index(all_chunks, save_path='index_v12.pkl')

    # --- Search with HyDE ---
    data = load_index('index_v12.pkl')
    model = get_embed_model()

    # Build train question embeddings for HyDE
    train_qs = train.drop_duplicates('question_id')
    train_texts_en = train_qs['question_en'].tolist()
    train_answers = train_qs['answer_en'].tolist()
    train_q_embs = model.encode(train_texts_en, batch_size=32, normalize_embeddings=True)
    log.info(f'Train question embeddings: {train_q_embs.shape}')

    # Load test queries
    queries = []
    with open('data/test.csv') as f:
        for row in csv.DictReader(f):
            queries.append({'query_id': row['query_id'], 'query': row['question']})
    log.info(f'Test queries: {len(queries)}')

    # Search
    PAD = 10.0
    fieldnames = ['query_id']
    for i in range(1, 6):
        fieldnames.extend([f'video_file_{i}', f'start_{i}', f'end_{i}'])

    rows = []
    hyde_used = 0

    for q in tqdm(queries, desc='Searching'):
        qv = model.encode([q['query']], normalize_embeddings=True)[0]

        # HyDE: find most similar train question, mix with its answer
        sim = train_q_embs @ qv
        best_idx = np.argmax(sim)
        best_sim = float(sim[best_idx])

        if best_sim > 0.7 and str(train_answers[best_idx]) != 'nan':
            answer_text = str(train_answers[best_idx])[:500]
            answer_emb = model.encode([answer_text], normalize_embeddings=True)[0]
            mixed = 0.6 * qv + 0.4 * answer_emb
            mixed = mixed / np.linalg.norm(mixed)
            search_vec = mixed
            hyde_used += 1
        else:
            search_vec = qv

        cands = search_index(search_vec, data, top_k=10)
        seen = set()
        results = []
        for c in cands:
            key = (c['video_file'], c.get('chunk_index', 0))
            if key in seen:
                continue
            seen.add(key)
            results.append(c)
            if len(results) >= 5:
                break

        row = {'query_id': q['query_id']}
        for i in range(5):
            si = str(i + 1)
            if i < len(results):
                r = results[i]
                vname = Path(r['video_file']).stem
                m = re.search(r'(video_[a-f0-9]+)', vname)
                vname = m.group(1) if m else vname
                row[f'video_file_{si}'] = vname
                row[f'start_{si}'] = round(max(0, r['start_time'] - PAD), 1)
                row[f'end_{si}'] = round(r['end_time'] + PAD, 1)
            else:
                row[f'video_file_{si}'] = 'video_02578eb3'
                row[f'start_{si}'] = 0.0
                row[f'end_{si}'] = 60.0
        rows.append(row)

    with open('submission_v12.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    log.info(f'HyDE used for {hyde_used}/{len(queries)} queries')
    log.info(f'Submission: submission_v12.csv ({len(rows)} rows)')
    log.info(f'Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
