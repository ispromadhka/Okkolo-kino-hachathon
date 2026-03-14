"""Search with dynamic HyDE and adaptive boundary prediction."""
import logging
import numpy as np
from pipeline.indexer import get_embed_model, load_index, search_index

log = logging.getLogger(__name__)
_index_data = None


def get_index(path='index_v19.pkl'):
    global _index_data
    if _index_data is None:
        _index_data = load_index(path)
        log.info(f'Index loaded: {len(_index_data["chunks"])} chunks')
    return _index_data


def search(query, top_k=5, train_q_embs=None, train_answers=None, p75=94.0):
    model = get_embed_model()
    data = get_index()

    qv = model.encode([query], normalize_embeddings=True)[0]

    # Dynamic HyDE
    search_vec = qv
    if train_q_embs is not None and train_answers is not None:
        sim = train_q_embs @ qv
        best_idx = np.argmax(sim)
        best_sim = float(sim[best_idx])
        if best_sim > 0.7 and str(train_answers[best_idx]) != 'nan':
            answer_emb = model.encode([str(train_answers[best_idx])[:500]], normalize_embeddings=True)[0]
            ans_weight = np.clip(0.2 + (best_sim - 0.7) * (0.5 / 0.3), 0.2, 0.7)
            mixed = (1.0 - ans_weight) * qv + ans_weight * answer_emb
            search_vec = mixed / np.linalg.norm(mixed)

    cands = search_index(search_vec, data, top_k=10)

    seen = set()
    results = []
    for c in cands:
        key = (c['video_file'], c.get('chunk_index', 0))
        if key in seen:
            continue
        seen.add(key)

        if c.get('chunk_type') == 'answer_aug':
            start = c['start_time']
            end = c['end_time']
        else:
            center = (c['start_time'] + c['end_time']) / 2.0
            start = max(0, center - p75 / 2.0)
            end = center + p75 / 2.0

        results.append({
            'video_file': c['video_file'],
            'start_time': start,
            'end_time': end,
            'score': c.get('score', 0),
            'chunk_type': c.get('chunk_type', ''),
        })
        if len(results) >= top_k:
            break

    return results
