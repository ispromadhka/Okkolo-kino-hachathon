import logging
from pipeline.indexer import get_embed_model, load_index, search_index

log = logging.getLogger(__name__)

_reranker = None
_index_data = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from flashrank import Ranker
        _reranker = Ranker(model_name='ms-marco-MultiBERT-L-12')
        log.info('FlashRank reranker loaded')
    return _reranker


def _get_index():
    global _index_data
    if _index_data is None:
        _index_data = load_index()
        log.info(f'Index loaded: {len(_index_data["chunks"])} chunks')
    return _index_data


def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Improved retriever:
    1. Search top-50 candidates (was 20) for better recall
    2. FlashRank reranking
    3. Dedup by (video_file, chunk_index)
    4. Smart output ordering:
       - Position 1: best reranker score (best for SR@1 and VR@1)
       - Positions 2+: greedily pick best score from unseen video if available,
         otherwise best remaining. This maximizes VR@K while preserving SR.
    """
    model = get_embed_model()
    data = _get_index()

    # Search top-50
    q_vec = model.encode([query], normalize_embeddings=True)[0]
    results = search_index(q_vec, data, top_k=50)

    if not results:
        return []

    # Rerank
    from flashrank import RerankRequest
    reranker = get_reranker()
    passages = [{'id': i, 'text': r['text'][:512]} for i, r in enumerate(results)]
    req = RerankRequest(query=query, passages=passages)
    ranked = reranker.rerank(req)

    # Build deduplicated candidate list (sorted by reranker score)
    candidates = []
    seen = set()
    for r in ranked:
        idx = r['id']
        c = results[idx]
        key = (c['video_file'], c.get('chunk_index', 0))
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            'video_file': c['video_file'],
            'start_time': c['start_time'],
            'end_time': c['end_time'],
            'score': r['score'],
            'text': c['text'][:200],
            'chunk_type': c.get('chunk_type', ''),
        })

    if not candidates:
        return []

    # Smart ordering: position 1 = best score, then greedily pick
    # best from unseen video, falling back to best remaining
    output = []
    used = set()
    seen_videos = set()

    # Position 1: absolute best
    output.append(candidates[0])
    used.add(0)
    seen_videos.add(candidates[0]['video_file'])

    # Remaining positions: prefer unseen videos, but if all remaining
    # are from seen videos, take next best
    for pos in range(1, top_k):
        # First try: best from unseen video
        best_new = None
        for i, c in enumerate(candidates):
            if i in used:
                continue
            if c['video_file'] not in seen_videos:
                best_new = i
                break

        # Second try: best remaining (any video)
        best_any = None
        for i, c in enumerate(candidates):
            if i in used:
                continue
            best_any = i
            break

        # Use new video if available, otherwise any
        pick = best_new if best_new is not None else best_any
        if pick is None:
            break

        output.append(candidates[pick])
        used.add(pick)
        seen_videos.add(candidates[pick]['video_file'])

    return output[:top_k]
