import logging
import numpy as np
from pipeline.indexer import get_embed_model, load_index, search_index

log = logging.getLogger(__name__)

_reranker = None
_index_data = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from flashrank import Ranker
        _reranker = Ranker(model_name='ms-marco-MultiBERT-L-12')
        log.info('FlashRank loaded')
    return _reranker


def get_index():
    global _index_data
    if _index_data is None:
        _index_data = load_index()
        log.info(f'Index loaded: {len(_index_data["chunks"])} chunks')
    return _index_data


def search(query, top_k=5, use_reranker=True):
    model = get_embed_model()
    data = get_index()

    q_vec = model.encode([query], normalize_embeddings=True)[0]
    candidates = search_index(q_vec, data, top_k=50)

    if not candidates:
        return []

    if use_reranker:
        try:
            from flashrank import RerankRequest
            reranker = get_reranker()
            passages = [{'id': i, 'text': c['text'][:512]} for i, c in enumerate(candidates)]
            ranked = reranker.rerank(RerankRequest(query=query, passages=passages))
            # Reorder candidates by reranker score
            reordered = []
            for r in ranked:
                idx = int(r['id'])  # fix: ensure int
                c = candidates[idx].copy()
                c['score'] = float(r['score'])
                reordered.append(c)
            candidates = reordered
        except Exception as e:
            log.warning(f'Reranker failed: {e}, using cosine scores')

    # Simple dedup by (video, chunk_index) — keep score order, no forced diversity
    output = []
    seen = set()
    for c in candidates:
        key = (c['video_file'], c.get('chunk_index', 0))
        if key in seen:
            continue
        seen.add(key)
        output.append({
            'video_file': c['video_file'],
            'start_time': c['start_time'],
            'end_time': c['end_time'],
            'score': c.get('score', 0),
            'text': c['text'][:200],
            'chunk_type': c.get('chunk_type', ''),
        })
        if len(output) >= top_k:
            break

    return output
