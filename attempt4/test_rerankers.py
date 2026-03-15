"""
Quick test: which rerankers load and work on this system?
Run this FIRST to determine which reranker to use in v27.

Usage: python test_rerankers.py [--device cuda:1]

Tests 3 reranker models with 3 loading approaches each.
Takes ~2-3 minutes total.
"""
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

DEVICE = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else 'cuda:1'
if '--device' in sys.argv:
    idx = sys.argv.index('--device')
    DEVICE = sys.argv[idx + 1]

TEST_QUERY = "Кто главный герой фильма?"
TEST_PASSAGES = [
    "Главный герой фильма — молодой человек по имени Алексей, который живёт в Москве.",
    "Погода в этот день была солнечной и тёплой, температура достигала 25 градусов.",
    "The main character is a young woman named Sarah who works as a detective.",
]

MODELS = {
    'bge': 'BAAI/bge-reranker-v2-m3',
    'jina': 'jinaai/jina-reranker-v2-base-multilingual',
    'gte': 'Alibaba-NLP/gte-reranker-modernbert-base',
}


def test_transformers(model_name: str):
    """Test loading via raw transformers."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    )
    model.eval().to(DEVICE)

    pairs = [[TEST_QUERY, p] for p in TEST_PASSAGES]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(DEVICE)
        scores = model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()
    return scores


def test_cross_encoder(model_name: str):
    """Test loading via sentence_transformers CrossEncoder."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, trust_remote_code=True, device=DEVICE)
    pairs = [[TEST_QUERY, p] for p in TEST_PASSAGES]
    scores = model.predict(pairs)
    return scores.tolist() if hasattr(scores, 'tolist') else list(scores)


def test_flagembedding_patched(model_name: str):
    """Test loading via FlagEmbedding with monkey-patch."""
    import transformers.utils.import_utils
    if not hasattr(transformers.utils.import_utils, 'is_torch_fx_available'):
        transformers.utils.import_utils.is_torch_fx_available = lambda: True
        import transformers.utils
        if not hasattr(transformers.utils, 'is_torch_fx_available'):
            transformers.utils.is_torch_fx_available = lambda: True

    from FlagEmbedding import FlagReranker
    reranker = FlagReranker(model_name, use_fp16=True, device=DEVICE)
    pairs = [[TEST_QUERY, p] for p in TEST_PASSAGES]
    scores = reranker.compute_score(pairs, normalize=False)
    if isinstance(scores, (int, float)):
        scores = [scores]
    return list(scores)


def main():
    log.info(f'Testing rerankers on device: {DEVICE}')
    log.info(f'Query: {TEST_QUERY}')
    log.info(f'Passages: {len(TEST_PASSAGES)}')
    log.info('')

    results = {}

    for short_name, model_name in MODELS.items():
        log.info(f'=== {short_name}: {model_name} ===')

        approaches = [
            ('transformers', test_transformers),
            ('CrossEncoder', test_cross_encoder),
            ('FlagEmbedding', test_flagembedding_patched),
        ]

        for approach_name, test_fn in approaches:
            try:
                t0 = time.time()
                scores = test_fn(model_name)
                elapsed = time.time() - t0
                log.info(f'  {approach_name}: OK ({elapsed:.1f}s)')
                log.info(f'    Scores: {[f"{s:.4f}" for s in scores]}')
                log.info(f'    Ranking: {sorted(range(len(scores)), key=lambda i: -scores[i])}')
                results[f'{short_name}/{approach_name}'] = 'OK'
                break  # First working approach is enough
            except Exception as e:
                log.warning(f'  {approach_name}: FAILED — {type(e).__name__}: {e}')
                results[f'{short_name}/{approach_name}'] = f'FAILED: {e}'

        # Free GPU memory
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        log.info('')

    log.info('=== SUMMARY ===')
    for k, v in results.items():
        status = 'OK' if v == 'OK' else 'FAILED'
        log.info(f'  {k}: {status}')


if __name__ == '__main__':
    main()
