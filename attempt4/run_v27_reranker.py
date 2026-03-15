"""
V27 pipeline — Cross-Encoder Reranker on top of V26 (0.572)

KEY INSIGHT: Cross-encoder rerankers typically improve retrieval by 30-40%.
Our FlagEmbedding crash was a LIBRARY BUG, not a technique problem.

Three approaches to load reranker (in order of preference):
  A) Raw transformers: AutoModelForSequenceClassification (no FlagEmbedding needed)
  B) sentence_transformers.CrossEncoder (clean API)
  C) FlagEmbedding with monkey-patch fix (is_torch_fx_available)

This script:
1. Builds same index as v19 (90s/30s windows + answer_aug)
2. Uses fine-tuned BGE-M3 for retrieval (top-20)
3. Cross-encoder reranks top-20 → top-5
4. Same boundary logic: answer_aug → exact GT, windows → center ± P75/2

Reranker options (set via --reranker flag):
  bge     → BAAI/bge-reranker-v2-m3 (best overall, multilingual)
  jina    → jinaai/jina-reranker-v2-base-multilingual (multilingual, up to 1024 tokens)
  gte     → Alibaba-NLP/gte-reranker-modernbert-base (newest, up to 8192 tokens)

Usage:
  python run_v27_reranker.py --reranker bge
  python run_v27_reranker.py --reranker jina
  python run_v27_reranker.py --reranker gte
  python run_v27_reranker.py --reranker bge --hyde-threshold 0.6 --top-rerank 20
"""
import logging
import pickle
import csv
import re
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ============================================================
# Reranker loader — 3 approaches, all avoid FlagEmbedding crash
# ============================================================

RERANKER_MODELS = {
    'bge': 'BAAI/bge-reranker-v2-m3',
    'jina': 'jinaai/jina-reranker-v2-base-multilingual',
    'gte': 'Alibaba-NLP/gte-reranker-modernbert-base',
}


def load_reranker_transformers(model_name: str, device: str = 'cuda'):
    """
    Load reranker using raw transformers (most reliable, no FlagEmbedding needed).
    Works for BGE, GTE. For Jina, may need trust_remote_code=True.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    log.info(f'Loading reranker via transformers: {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    model.to(device)
    log.info(f'Reranker loaded on {device}')

    def rerank_fn(query: str, passages: list[str], batch_size: int = 16) -> list[float]:
        """Score query-passage pairs. Returns list of float scores."""
        all_scores = []
        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i:i + batch_size]
            pairs = [[query, p] for p in batch_passages]
            with torch.no_grad():
                inputs = tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512,
                ).to(device)
                logits = model(**inputs, return_dict=True).logits.view(-1).float()
                all_scores.extend(logits.cpu().tolist())
        return all_scores

    return rerank_fn


def load_reranker_cross_encoder(model_name: str, device: str = 'cuda'):
    """
    Load reranker using sentence_transformers CrossEncoder.
    Cleaner API but may not support all models.
    """
    from sentence_transformers import CrossEncoder

    log.info(f'Loading reranker via CrossEncoder: {model_name}')
    model = CrossEncoder(
        model_name,
        trust_remote_code=True,
        device=device,
    )
    log.info(f'CrossEncoder loaded on {device}')

    def rerank_fn(query: str, passages: list[str], batch_size: int = 16) -> list[float]:
        pairs = [[query, p] for p in passages]
        scores = model.predict(pairs, batch_size=batch_size)
        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)

    return rerank_fn


def load_reranker_flagembedding_patched(model_name: str, device: str = 'cuda'):
    """
    Load reranker via FlagEmbedding WITH monkey-patch for the
    is_torch_fx_available crash (transformers >= 5.0 removed it).
    """
    # Monkey-patch BEFORE importing FlagEmbedding
    import transformers.utils.import_utils
    if not hasattr(transformers.utils.import_utils, 'is_torch_fx_available'):
        transformers.utils.import_utils.is_torch_fx_available = lambda: True
        # Also patch it in transformers.utils namespace
        if not hasattr(transformers.utils, 'is_torch_fx_available'):
            transformers.utils.is_torch_fx_available = lambda: True
        log.info('Monkey-patched is_torch_fx_available → True')

    from FlagEmbedding import FlagReranker

    log.info(f'Loading reranker via FlagReranker: {model_name}')
    reranker = FlagReranker(model_name, use_fp16=True, device=device)
    log.info(f'FlagReranker loaded on {device}')

    def rerank_fn(query: str, passages: list[str], batch_size: int = 16) -> list[float]:
        pairs = [[query, p] for p in passages]
        scores = reranker.compute_score(pairs, normalize=False)
        if isinstance(scores, (int, float)):
            scores = [scores]
        return list(scores)

    return rerank_fn


def load_reranker(name: str, device: str = 'cuda'):
    """Try loading reranker with multiple fallback approaches."""
    model_name = RERANKER_MODELS.get(name, name)

    # Approach A: raw transformers (most reliable)
    try:
        return load_reranker_transformers(model_name, device)
    except Exception as e:
        log.warning(f'Transformers loading failed: {e}')

    # Approach B: sentence_transformers CrossEncoder
    try:
        return load_reranker_cross_encoder(model_name, device)
    except Exception as e:
        log.warning(f'CrossEncoder loading failed: {e}')

    # Approach C: FlagEmbedding with monkey-patch
    try:
        return load_reranker_flagembedding_patched(model_name, device)
    except Exception as e:
        log.warning(f'FlagEmbedding loading failed: {e}')

    raise RuntimeError(f'All reranker loading approaches failed for {model_name}')


# ============================================================
# Main pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='V27: Cross-encoder reranker pipeline')
    parser.add_argument('--reranker', type=str, default='bge',
                        choices=['bge', 'jina', 'gte'],
                        help='Reranker model to use')
    parser.add_argument('--reranker-device', type=str, default='cuda:1',
                        help='Device for reranker (use different GPU from embedder)')
    parser.add_argument('--embed-device', type=str, default='cuda:0',
                        help='Device for embedding model')
    parser.add_argument('--hyde-threshold', type=float, default=0.6,
                        help='HyDE similarity threshold (v26 uses 0.6)')
    parser.add_argument('--top-rerank', type=int, default=20,
                        help='Number of candidates to rerank')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Final number of results per query')
    parser.add_argument('--skip-index', action='store_true',
                        help='Skip index building (reuse existing index_v27.pkl)')
    parser.add_argument('--index-path', type=str, default='index_v27.pkl',
                        help='Path to index file')
    parser.add_argument('--embed-model', type=str, default=None,
                        help='Path to fine-tuned embedding model (default: BAAI/bge-m3)')
    parser.add_argument('--window-sec', type=float, default=90.0,
                        help='Window size in seconds')
    parser.add_argument('--overlap-sec', type=float, default=30.0,
                        help='Window overlap in seconds')
    parser.add_argument('--gt-sim-threshold', type=float, default=0.95,
                        help='Similarity threshold for direct GT return')
    parser.add_argument('--use-vector-prf', action='store_true', default=True,
                        help='Use Vector PRF for non-HyDE queries')
    parser.add_argument('--no-vector-prf', action='store_true',
                        help='Disable Vector PRF')
    parser.add_argument('--rerank-batch', type=int, default=16,
                        help='Reranker batch size')
    args = parser.parse_args()

    if args.no_vector_prf:
        args.use_vector_prf = False

    t0 = time.time()

    # --- Load embedding model ---
    from sentence_transformers import SentenceTransformer

    embed_model_name = args.embed_model or 'BAAI/bge-m3'
    log.info(f'Loading embedding model: {embed_model_name}')
    embed_model = SentenceTransformer(embed_model_name, device=args.embed_device)
    log.info('Embedding model loaded')

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

    # --- Load train data ---
    import pandas as pd
    train = pd.read_csv('data/train_qa.csv')
    P75 = (train['end'] - train['start']).quantile(0.75)
    log.info(f'P75 fragment length: {P75:.1f}s')

    # --- Build chunks ---
    if not args.skip_index:
        from pipeline.chunker import merge_segments_to_window

        all_chunks = []

        # 1. Window chunks
        for key, segs in transcripts.items():
            if not segs:
                continue
            m = re.search(r'(video_[a-f0-9]+)', key)
            vid = m.group(1) if m else key
            vfile = video_map.get(vid, f'videos/{vid}.mp4')
            windows = merge_segments_to_window(segs, args.window_sec, args.overlap_sec)
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

        # 3. Embed all chunks
        log.info(f'Encoding {len(all_chunks)} chunks...')
        texts = [c['text'] for c in all_chunks]
        embeddings = embed_model.encode(
            texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
        )
        data = {'embeddings': embeddings, 'chunks': all_chunks}
        with open(args.index_path, 'wb') as f:
            pickle.dump(data, f)
        log.info(f'Index saved: {args.index_path} ({len(all_chunks)} chunks)')
    else:
        log.info(f'Loading existing index: {args.index_path}')
        with open(args.index_path, 'rb') as f:
            data = pickle.load(f)
        log.info(f'Index loaded: {len(data["chunks"])} chunks')

    # --- Load reranker ---
    rerank_fn = load_reranker(args.reranker, device=args.reranker_device)

    # --- Prepare train questions for HyDE ---
    train_qs = train.drop_duplicates('question_id')
    train_texts_en = train_qs['question_en'].tolist()
    train_answers = train_qs['answer_en'].tolist()
    train_q_embs = embed_model.encode(train_texts_en, batch_size=32, normalize_embeddings=True)

    # Also load train ground truth for direct GT return
    train_gt = {}
    for _, row in train.iterrows():
        qid = row['question_id']
        if qid not in train_gt:
            train_gt[qid] = {
                'video_file': row['video_file'],
                'start': float(row['start']),
                'end': float(row['end']),
                'question_en': row['question_en'],
            }

    log.info(f'Train question embeddings: {train_q_embs.shape}')

    # --- Load test queries ---
    queries = []
    with open('data/test.csv') as f:
        for row in csv.DictReader(f):
            queries.append({'query_id': row['query_id'], 'query': row['question']})
    log.info(f'Test queries: {len(queries)}')

    # --- Search + Rerank ---
    fieldnames = ['query_id']
    for i in range(1, 6):
        fieldnames.extend([f'video_file_{i}', f'start_{i}', f'end_{i}'])

    rows = []
    hyde_used = 0
    gt_direct = 0
    prf_used = 0

    stats = {'sim_bins': {}}

    for q in tqdm(queries, desc='Search+Rerank'):
        qv = embed_model.encode([q['query']], normalize_embeddings=True)[0]

        # --- Dynamic HyDE ---
        sim = train_q_embs @ qv
        best_idx = np.argmax(sim)
        best_sim = float(sim[best_idx])

        # Track similarity distribution
        bin_key = f'{best_sim:.1f}'
        stats['sim_bins'][bin_key] = stats['sim_bins'].get(bin_key, 0) + 1

        # Direct GT return for very high similarity
        if best_sim > args.gt_sim_threshold:
            gt_qid = train_qs.iloc[best_idx]['question_id']
            if gt_qid in train_gt:
                gt = train_gt[gt_qid]
                gt_direct += 1
                row = {'query_id': q['query_id']}
                vname = Path(gt['video_file']).stem
                m = re.search(r'(video_[a-f0-9]+)', vname)
                vname = m.group(1) if m else vname
                for si in range(1, 6):
                    row[f'video_file_{si}'] = vname
                    row[f'start_{si}'] = round(gt['start'], 1)
                    row[f'end_{si}'] = round(gt['end'], 1)
                rows.append(row)
                continue

        # HyDE mixing
        if best_sim > args.hyde_threshold and str(train_answers[best_idx]) != 'nan':
            answer_text = str(train_answers[best_idx])[:500]
            answer_emb = embed_model.encode([answer_text], normalize_embeddings=True)[0]
            # Dynamic weight based on similarity
            weight_range = min(0.3, 1.0 - args.hyde_threshold)  # avoid div by zero
            ans_weight = np.clip(
                0.2 + (best_sim - args.hyde_threshold) * (0.5 / weight_range),
                0.2, 0.7
            )
            query_weight = 1.0 - ans_weight
            mixed = query_weight * qv + ans_weight * answer_emb
            search_vec = mixed / np.linalg.norm(mixed)
            hyde_used += 1
        elif args.use_vector_prf and best_sim < args.hyde_threshold:
            # Vector PRF: use top-3 initial results to refine query
            initial_scores = data['embeddings'] @ qv
            top3_idx = np.argsort(initial_scores)[::-1][:3]
            prf_vecs = data['embeddings'][top3_idx]
            prf_centroid = np.mean(prf_vecs, axis=0)
            prf_centroid = prf_centroid / np.linalg.norm(prf_centroid)
            # Mix: 80% query + 20% PRF centroid
            search_vec = 0.8 * qv + 0.2 * prf_centroid
            search_vec = search_vec / np.linalg.norm(search_vec)
            prf_used += 1
        else:
            search_vec = qv

        # --- Initial retrieval: top-N candidates ---
        scores = data['embeddings'] @ search_vec
        top_idx = np.argsort(scores)[::-1][:args.top_rerank]

        candidates = []
        for idx in top_idx:
            chunk = data['chunks'][idx].copy()
            chunk['embed_score'] = float(scores[idx])
            candidates.append(chunk)

        # --- Cross-encoder reranking ---
        if candidates:
            passages = [c['text'] for c in candidates]
            try:
                rerank_scores = rerank_fn(q['query'], passages, batch_size=args.rerank_batch)
                for i, c in enumerate(candidates):
                    c['rerank_score'] = rerank_scores[i]
                # Sort by rerank score (higher is better)
                candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
            except Exception as e:
                log.warning(f'Reranking failed for query {q["query_id"]}: {e}')
                # Fallback: keep embedding-based order
                candidates.sort(key=lambda x: x['embed_score'], reverse=True)

        # --- Dedup and select top-k ---
        seen = set()
        results = []
        for c in candidates:
            key = (c['video_file'], c.get('chunk_index', 0))
            if key in seen:
                continue
            seen.add(key)
            results.append(c)
            if len(results) >= args.top_k:
                break

        # --- Build output row ---
        row = {'query_id': q['query_id']}
        for i in range(5):
            si = str(i + 1)
            if i < len(results):
                r = results[i]
                vname = Path(r['video_file']).stem
                m = re.search(r'(video_[a-f0-9]+)', vname)
                vname = m.group(1) if m else vname

                if r.get('chunk_type') == 'answer_aug':
                    start = r['start_time']
                    end = r['end_time']
                else:
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

    # --- Write submission ---
    out_path = f'submission_v27_{args.reranker}.csv'
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # --- Stats ---
    log.info(f'=== V27 Results ({args.reranker} reranker) ===')
    log.info(f'HyDE used: {hyde_used}/{len(queries)} ({hyde_used/len(queries)*100:.0f}%)')
    log.info(f'Direct GT: {gt_direct}/{len(queries)} ({gt_direct/len(queries)*100:.0f}%)')
    log.info(f'Vector PRF: {prf_used}/{len(queries)} ({prf_used/len(queries)*100:.0f}%)')
    log.info(f'Similarity distribution:')
    for k in sorted(stats['sim_bins'].keys()):
        log.info(f'  sim={k}: {stats["sim_bins"][k]} queries')
    log.info(f'Submission: {out_path} ({len(rows)} rows)')
    log.info(f'Total: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
