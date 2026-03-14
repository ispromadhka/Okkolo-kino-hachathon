"""
V19 pipeline — score 0.518 (1st place)

Key improvements over v12 (0.500):
1. Dynamic HyDE weight: more similar train question → more answer influence
2. Zero padding for answer_aug chunks (exact GT timestamps)
3. P75 boundary truncation for window chunks (center ± 47s)

Pipeline:
1. Build 90s/30s window chunks from faster-whisper large-v3-turbo transcripts
2. Add answer_en from train as augmentation chunks (exact timestamps)
3. Embed all with BGE-M3 (1024d, normalized) → numpy index
4. For each test query:
   a. Find most similar train question (cosine on BGE-M3)
   b. Dynamic HyDE: weight = 0.2 + (sim-0.7)*(0.5/0.3), clamped [0.2, 0.7]
   c. Mix query embedding with answer embedding using dynamic weight
   d. Cosine search top-10 → dedup
   e. Boundary: answer_aug → exact timestamps, windows → center ± P75/2
5. Output: 5 × (video_stem, start, end)
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

    # 1. Window chunks (90s/30s)
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

    # 2. Answer augmentation
    import pandas as pd
    train = pd.read_csv('data/train_qa.csv')
    P75 = (train['end'] - train['start']).quantile(0.75)
    log.info(f'P75 fragment length: {P75:.1f}s')

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
    log.info(f'Answer augmentation: +{aug_count} chunks, total: {len(all_chunks)}')

    # 3. Embed and save
    build_index(all_chunks, save_path='index_v19.pkl')

    # --- Search with Dynamic HyDE ---
    data = load_index('index_v19.pkl')
    model = get_embed_model()

    train_qs = train.drop_duplicates('question_id')
    train_texts_en = train_qs['question_en'].tolist()
    train_answers = train_qs['answer_en'].tolist()
    train_q_embs = model.encode(train_texts_en, batch_size=32, normalize_embeddings=True)
    log.info(f'Train question embeddings: {train_q_embs.shape}')

    queries = []
    with open('data/test.csv') as f:
        for row in csv.DictReader(f):
            queries.append({'query_id': row['query_id'], 'query': row['question']})
    log.info(f'Test queries: {len(queries)}')

    fieldnames = ['query_id']
    for i in range(1, 6):
        fieldnames.extend([f'video_file_{i}', f'start_{i}', f'end_{i}'])

    rows = []
    hyde_used = 0

    for q in tqdm(queries, desc='Searching'):
        qv = model.encode([q['query']], normalize_embeddings=True)[0]

        # Dynamic HyDE
        sim = train_q_embs @ qv
        best_idx = np.argmax(sim)
        best_sim = float(sim[best_idx])

        if best_sim > 0.7 and str(train_answers[best_idx]) != 'nan':
            answer_text = str(train_answers[best_idx])[:500]
            answer_emb = model.encode([answer_text], normalize_embeddings=True)[0]
            # Dynamic weight: more similar → more answer influence
            ans_weight = np.clip(0.2 + (best_sim - 0.7) * (0.5 / 0.3), 0.2, 0.7)
            query_weight = 1.0 - ans_weight
            mixed = query_weight * qv + ans_weight * answer_emb
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

                if r.get('chunk_type') == 'answer_aug':
                    # Zero padding: trust exact GT timestamps
                    start = r['start_time']
                    end = r['end_time']
                else:
                    # Window chunks: center ± P75/2
                    center = (r['start_time'] + r['end_time']) / 2.0
                    start = max(0, center - P75 / 2.0)
                    end = center + P75 / 2.0

                row[f'video_file_{si}'] = vname
                row[f'start_{si}'] = round(start, 1)
                row[f'end_{si}'] = round(end, 1)
            else:
                row[f'video_file_{si}'] = 'video_02578eb3'
                row[f'start_{si}'] = 0.0
                row[f'end_{si}'] = 60.0
        rows.append(row)

    with open('submission_v19.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    log.info(f'HyDE used: {hyde_used}/{len(queries)} ({hyde_used/len(queries)*100:.0f}%)')
    log.info(f'Submission: submission_v19.csv ({len(rows)} rows)')
    log.info(f'Total: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
