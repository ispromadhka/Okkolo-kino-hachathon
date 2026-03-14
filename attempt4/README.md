# Attempt 4: Dynamic HyDE + Zero Padding — Score 0.518 (1st place)

## Score progression

| Version | Score | Key change |
|---------|-------|------------|
| v4 | 0.367 | Baseline: Whisper Tiny + BGE-M3 + 90s windows + ±10s padding |
| v8 | 0.377 | faster-whisper large-v3-turbo retranscription |
| v11 | 0.470 | Answer augmentation from train data (+0.09) |
| v12 | 0.500 | HyDE query expansion (+0.03) |
| v15 | 0.508 | P75 boundary truncation for windows, ±5s for answers |
| **v19** | **0.518** | **Dynamic HyDE weight + zero padding for answers** |

## What makes v19 the best

### 1. Dynamic HyDE weight
Instead of fixed 0.6/0.4 mixing of query and answer embeddings:
```
ans_weight = clamp(0.2 + (similarity - 0.7) * (0.5 / 0.3), 0.2, 0.7)
search_vec = (1 - ans_weight) * query_emb + ans_weight * answer_emb
```
- similarity = 0.7 (threshold) → ans_weight = 0.2 (mostly query)
- similarity = 0.85 → ans_weight = 0.45 (balanced)
- similarity = 1.0 → ans_weight = 0.7 (mostly answer)

More confident train match → trust answer embedding more.

### 2. Zero padding for answer_aug chunks
Answer augmentation chunks already have exact ground truth timestamps from training data. Adding padding (±5s or ±10s) only reduces IoU. Zero padding = maximum IoU.

### 3. P75 boundary truncation for windows
Window chunks (90s) are too wide for IoU ≥ 0.5. Center the prediction and trim to P75 of train fragment distribution (94s). This matches the typical ground truth fragment length.

## Architecture

```
OFFLINE:
  Audio (436 files)
    → faster-whisper large-v3-turbo (8 GPU parallel, beam_size=5)
    → new_transcripts.pkl

  Transcripts → 90s/30s sliding windows → 5010 chunks
  Train answers → answer augmentation → +4466 chunks
  Total: 9476 chunks → BGE-M3 1024d → numpy index

  Train questions → BGE-M3 → train question index (for HyDE matching)

ONLINE (<100ms/query):
  Query → BGE-M3 encode
  → Find similar train question (cosine)
  → Dynamic HyDE: mix with answer embedding (variable weight)
  → Cosine search top-10 → dedup
  → Adaptive boundary:
      answer_aug → exact timestamps (zero padding)
      windows → center ± P75/2 (47s each side)
  → 5 × (video_stem, start, end)
```

## What didn't work (lessons learned)

| Approach | Score | Why it failed |
|----------|-------|---------------|
| Multi-scale 171K chunks | 0.289 | Too many sentence chunks flooded results |
| Cascaded coarse+fine | 0.328 | Lost video diversity in top-5 |
| bge-reranker-v2-m3 | 0.484 | Cross-encoder re-ordered away from best temporal matches |
| question_en/ru augmentation | 0.477 | Questions too similar, duplicated chunks |
| Hybrid dense+sparse RRF | 0.404 | Sparse search disrupted ranking |
| Dynamic boundary by score | 0.508 | Score doesn't correlate with optimal window width on test |

## Key insight

The biggest gains came from **bridging the semantic gap** between queries and transcripts:
- Answer augmentation (+0.09): answers use query-like vocabulary
- HyDE (+0.03): shifts search vector toward answer space
- Dynamic HyDE (+0.01): better calibration of mixing weights

Chunking and boundary strategies gave smaller gains (+0.008-0.018). The retrieval quality matters more than boundary precision.

## Files

```
attempt4/
├── run_v19.py                 # Complete pipeline (score 0.518)
├── config.py                  # Parameters
├── evaluate.py                # Local evaluation (SR@K, VR@K)
├── retranscribe_parallel.py   # 8-GPU parallel ASR
├── retranscribe_fw.py         # Single-GPU ASR fallback
├── merge_transcripts.py       # Merge per-GPU transcripts
├── pipeline/
│   ├── chunker.py             # Sliding window chunker
│   └── indexer.py             # BGE-M3 numpy index
└── search/
    └── retriever.py           # Dynamic HyDE + adaptive boundary
```

## Usage

```bash
# Symlink data
ln -sf /path/to/data data
ln -sf /path/to/new_transcripts.pkl new_transcripts.pkl

# Run pipeline
python run_v19.py
# Output: submission_v19.csv

# Evaluate locally
python evaluate.py --sample 200
```

## Requirements
```
sentence-transformers
faster-whisper
numpy
pandas
tqdm
static-ffmpeg
```
