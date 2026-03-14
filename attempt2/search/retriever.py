"""
Retriever with adaptive padding per chunk scale.

Key insight: padding must match chunk granularity.
- Sentence chunks (3-10s) → pad ±8s → predicted window ~20-26s
- Group3 chunks (10-20s) → pad ±6s → predicted window ~22-32s
- Short windows (20s) → pad ±4s → predicted window ~28s
- Medium windows (45s) → pad ±2s → predicted window ~49s
- Large windows (90s) → pad ±0s → predicted window ~90s
- Summary (full video) → pad 0, but deprioritize for SR@K
"""
import logging
import numpy as np
from pipeline.indexer import get_embed_model, load_index, search_index

log = logging.getLogger(__name__)

_reranker = None
_index_data = None


def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from flashrank import Ranker
            _reranker = Ranker(model_name='ms-marco-MultiBERT-L-12')
            log.info('FlashRank loaded')
        except Exception as e:
            log.warning(f'FlashRank not available: {e}')
    return _reranker


def get_index():
    global _index_data
    if _index_data is None:
        _index_data = load_index()
        log.info(f'Index loaded: {len(_index_data["chunks"])} chunks')
    return _index_data


def search(query, top_k=5, use_reranker=False):
    model = get_embed_model()
    data = get_index()

    q_vec = model.encode([query], normalize_embeddings=True)[0]
    candidates = search_index(q_vec, data, top_k=50)

    if not candidates:
        return []

    # Optional reranking
    if use_reranker:
        reranker = get_reranker()
        if reranker:
            try:
                from flashrank import RerankRequest
                passages = [{'id': i, 'text': c['text'][:512]} for i, c in enumerate(candidates)]
                ranked = reranker.rerank(RerankRequest(query=query, passages=passages))
                reordered = []
                for r in ranked:
                    idx = int(r['id'])
                    c = candidates[idx].copy()
                    c['score'] = float(r['score'])
                    reordered.append(c)
                candidates = reordered
            except Exception as e:
                log.warning(f'Reranker failed: {e}')

    # Dedup by (video_file, chunk_index) — keep score order
    # But deprioritize summary chunks (they kill SR@K with huge windows)
    output_primary = []
    output_summary = []
    seen = set()

    for c in candidates:
        key = (c['video_file'], c.get('chunk_index', 0))
        if key in seen:
            continue
        seen.add(key)

        if c.get('scale') == 'summary':
            output_summary.append(c)
        else:
            output_primary.append(c)

    # Fill top_k: primary results first, then summary as fallback
    output = output_primary[:top_k]
    if len(output) < top_k:
        output.extend(output_summary[:top_k - len(output)])

    # Apply adaptive padding per chunk
    results = []
    for c in output[:top_k]:
        pad = c.get('padding', 5.0)
        results.append({
            'video_file': c['video_file'],
            'start_time': max(0, c['start_time'] - pad),
            'end_time': c['end_time'] + pad,
            'score': c.get('score', 0),
            'text': c['text'][:200],
            'chunk_type': c.get('chunk_type', ''),
            'scale': c.get('scale', ''),
        })

    return results
