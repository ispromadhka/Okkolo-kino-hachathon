import numpy as np
import pickle
import logging

log = logging.getLogger(__name__)

_embed_model = None
INDEX_PATH = 'index_data.pkl'

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer('BAAI/bge-m3')
        log.info('BGE-M3 loaded')
    return _embed_model

def build_index(chunks, save_path=INDEX_PATH):
    model = get_embed_model()
    texts = [c['text'] for c in chunks]
    log.info(f'Encoding {len(texts)} chunks...')
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    data = {'embeddings': embeddings, 'chunks': chunks}
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    log.info(f'Index saved: {save_path} ({len(chunks)} chunks)')
    return data

def load_index(path=INDEX_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)

def search_index(query_vec, data, top_k=20):
    scores = data['embeddings'] @ query_vec
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_idx:
        chunk = data['chunks'][idx].copy()
        chunk['score'] = float(scores[idx])
        results.append(chunk)
    return results
