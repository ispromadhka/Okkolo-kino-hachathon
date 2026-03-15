"""
FINAL SOLUTION — Score 0.572 (1st place)

Pipeline:
1. faster-whisper large-v3-turbo retranscription (8 GPU parallel)
2. Fine-tune BGE-M3 on 1812 train EN question-chunk pairs (4 epochs, MNR loss)
3. 90s/30s sliding window chunks + answer_en augmentation from train
4. Dynamic HyDE query expansion (threshold 0.6, variable weight)
5. Direct GT return for near-exact train matches (sim > 0.95)
6. Vector PRF for queries without HyDE match
7. Boundary: answer_aug → exact timestamps, windows → center ± P75/2
"""
import pickle, csv, re, numpy as np, time, os
os.environ["WANDB_DISABLED"] = "true"
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def fine_tune_model(train_pairs, base_model="BAAI/bge-m3", epochs=4, batch_size=16,
                    lr=2e-5, save_path="bge-m3-finetuned"):
    """Fine-tune BGE-M3 on question-chunk pairs with MNR loss."""
    log.info(f'Fine-tuning {base_model} on {len(train_pairs)} pairs, {epochs} epochs')
    model = SentenceTransformer(base_model)
    examples = [InputExample(texts=[q, p]) for q, p in train_pairs]
    dl = DataLoader(examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(dl, loss)],
        epochs=epochs,
        warmup_steps=50,
        show_progress_bar=True,
        optimizer_params={"lr": lr},
    )
    model.save(save_path)
    log.info(f'Model saved to {save_path}')
    return model


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
    log.info(f'Loaded {len(transcripts)} videos')

    from pipeline.chunker import merge_segments_to_window

    # --- Build chunks ---
    chunks_by_video = {}
    all_chunks = []
    for key, segs in transcripts.items():
        if not segs:
            continue
        m = re.search(r'(video_[a-f0-9]+)', key)
        vid = m.group(1) if m else key
        vfile = video_map.get(vid, f'videos/{vid}.mp4')
        windows = merge_segments_to_window(segs, 90.0, 30.0)
        chunks_by_video[vfile] = windows
        for i, w in enumerate(windows):
            all_chunks.append({
                'video_file': vfile, 'start_time': w['start'], 'end_time': w['end'],
                'text': w['text'], 'chunk_index': i, 'chunk_type': 'window',
            })

    train = pd.read_csv('data/train_qa.csv')
    P75 = (train['end'] - train['start']).quantile(0.75)
    log.info(f'P75 fragment length: {P75:.1f}s')

    # Answer augmentation
    aug = 0
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
            'chunk_index': 90000 + aug,
            'chunk_type': 'answer_aug',
        })
        aug += 1
    log.info(f'Chunks: {len(all_chunks)} ({len(all_chunks)-aug} windows + {aug} answers)')

    # --- Fine-tune BGE-M3 ---
    train_pairs = []
    for _, row in train.drop_duplicates('question_id').iterrows():
        vf = row['video_file']
        if vf not in chunks_by_video:
            continue
        best = max(
            chunks_by_video[vf],
            key=lambda w: max(0, min(float(row['end']), w['end']) - max(float(row['start']), w['start'])),
            default=None,
        )
        if not best:
            continue
        overlap = max(0, min(float(row['end']), best['end']) - max(float(row['start']), best['start']))
        if overlap < 10:
            continue
        train_pairs.append((str(row['question_en']), best['text'][:512]))

    model_path = "bge-m3-finetuned"
    if os.path.exists(model_path):
        log.info(f'Loading existing fine-tuned model from {model_path}')
        model = SentenceTransformer(model_path)
    else:
        model = fine_tune_model(train_pairs, epochs=4, save_path=model_path)

    # --- Embed chunks ---
    log.info('Encoding chunks...')
    embs = model.encode(
        [c['text'] for c in all_chunks],
        batch_size=32, show_progress_bar=True, normalize_embeddings=True,
    )
    log.info(f'Embeddings: {embs.shape}')

    # --- HyDE setup ---
    train_qs = train.drop_duplicates('question_id')
    tq_embs = model.encode(train_qs['question_en'].tolist(), batch_size=32, normalize_embeddings=True)
    tq_ans = train_qs['answer_en'].tolist()
    tq_ids = train_qs['question_id'].tolist()

    gt_by_qid = defaultdict(list)
    for _, row in train.iterrows():
        gt_by_qid[int(row['question_id'])].append({
            'video_file': row['video_file'],
            'start': float(row['start']),
            'end': float(row['end']),
        })

    # --- Load test queries ---
    queries = []
    with open('data/test.csv') as f:
        for row in csv.DictReader(f):
            queries.append({'query_id': row['query_id'], 'query': row['question']})
    log.info(f'Test queries: {len(queries)}')

    # --- Search ---
    fieldnames = ['query_id']
    for i in range(1, 6):
        fieldnames.extend([f'video_file_{i}', f'start_{i}', f'end_{i}'])

    rows = []
    direct_gt = 0
    hyde_used = 0
    prf_used = 0

    for q in tqdm(queries, desc='Searching'):
        qv = model.encode([q['query']], normalize_embeddings=True)[0]
        sim = tq_embs @ qv
        best_idx = np.argmax(sim)
        best_sim = float(sim[best_idx])

        # Direct GT return for near-exact matches
        if best_sim > 0.95:
            qid = tq_ids[best_idx]
            gts = gt_by_qid[int(qid)]
            direct_gt += 1
            row = {'query_id': q['query_id']}
            for i in range(5):
                si = str(i + 1)
                if i < len(gts):
                    g = gts[i]
                    vn = Path(g['video_file']).stem
                    m2 = re.search(r'(video_[a-f0-9]+)', vn)
                    row[f'video_file_{si}'] = m2.group(1) if m2 else vn
                    row[f'start_{si}'] = round(g['start'], 1)
                    row[f'end_{si}'] = round(g['end'], 1)
                else:
                    g = gts[0]
                    vn = Path(g['video_file']).stem
                    m2 = re.search(r'(video_[a-f0-9]+)', vn)
                    row[f'video_file_{si}'] = m2.group(1) if m2 else vn
                    row[f'start_{si}'] = 0.0
                    row[f'end_{si}'] = 60.0
            rows.append(row)
            continue

        # Dynamic HyDE
        if best_sim > 0.6 and str(tq_ans[best_idx]) != 'nan':
            answer_emb = model.encode([str(tq_ans[best_idx])[:500]], normalize_embeddings=True)[0]
            ans_weight = np.clip(0.1 + (best_sim - 0.6) * (0.6 / 0.4), 0.1, 0.7)
            search_vec = (1 - ans_weight) * qv + ans_weight * answer_emb
            search_vec = search_vec / np.linalg.norm(search_vec)
            hyde_used += 1
        else:
            # Vector PRF
            top3 = np.argsort(embs @ qv)[::-1][:3]
            centroid = embs[top3].mean(0)
            centroid = centroid / np.linalg.norm(centroid)
            search_vec = 0.7 * qv + 0.3 * centroid
            search_vec = search_vec / np.linalg.norm(search_vec)
            prf_used += 1

        # Cosine search
        scores = embs @ search_vec
        top_idx = np.argsort(scores)[::-1][:10]
        seen = set()
        results = []
        for idx in top_idx:
            c = all_chunks[idx]
            key = (c['video_file'], c.get('chunk_index', 0))
            if key in seen:
                continue
            seen.add(key)
            results.append(c)
            if len(results) >= 5:
                break

        # Format output with adaptive boundary
        row = {'query_id': q['query_id']}
        for i in range(5):
            si = str(i + 1)
            if i < len(results):
                r = results[i]
                vn = Path(r['video_file']).stem
                m2 = re.search(r'(video_[a-f0-9]+)', vn)
                vn = m2.group(1) if m2 else vn

                if r.get('chunk_type') == 'answer_aug':
                    start = r['start_time']
                    end = r['end_time']
                else:
                    center = (r['start_time'] + r['end_time']) / 2
                    start = max(0, center - P75 / 2)
                    end = center + P75 / 2

                row[f'video_file_{si}'] = vn
                row[f'start_{si}'] = round(start, 1)
                row[f'end_{si}'] = round(end, 1)
            else:
                row[f'video_file_{si}'] = 'video_02578eb3'
                row[f'start_{si}'] = 0.0
                row[f'end_{si}'] = 60.0
        rows.append(row)

    with open('submission.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    log.info(f'Direct GT: {direct_gt}, HyDE: {hyde_used}, PRF: {prf_used}')
    log.info(f'Submission: submission.csv ({len(rows)} rows)')
    log.info(f'Total: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
