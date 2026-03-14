# Attempt 2: Multi-scale chunking + Adaptive padding + Better ASR

## Score: attempt1 → 0.367, attempt2 → TBD (target: 0.45+)

## What changed from attempt1

### Problem diagnosis
Attempt1 used 90s sliding windows with fixed ±10s padding. This killed SR@K (Success Rate) because:
- If ground truth is 15s and our window is 90s → IoU = 15/90 = 0.17 (fails IoU ≥ 0.5)
- Fixed padding doesn't adapt to chunk granularity
- Summary chunks (full video) polluted top results for SR

### Key improvements

#### 1. Multi-scale chunking (`pipeline/chunker.py`)
Instead of one window size, we index at 5 granularities simultaneously:

| Scale | Size | Padding | Purpose |
|-------|------|---------|---------|
| `sentence` | 3-10s (one ASR segment) | ±8s | Catch short GT fragments, maximize IoU |
| `group3` | 10-20s (3 segments) | ±6s | Medium precision, natural speech boundaries |
| `short` | 20s window / 10s overlap | ±4s | Short context matches |
| `medium` | 45s window / 15s overlap | ±2s | Medium context matches |
| `large` | 90s window / 30s overlap | ±0s | Long context + VR@K |

Plus one `summary` chunk per video (full text, truncated to 3000 chars) for Video Recall.

#### 2. Adaptive padding (`search/retriever.py`)
Each chunk carries its own `padding` value based on its scale. The retriever applies padding per-chunk:
- Sentence (3-10s) → padded to ~20-26s → IoU with 15s GT ≈ 0.5+
- Short window (20s) → padded to ~28s → IoU with 30s GT ≈ 0.6+
- Large window (90s) → no padding → covers broad context

#### 3. Smart result ordering
Summary chunks are deprioritized (moved to positions 4-5 only as fallback). Primary results come from sentence/group/window chunks which have better IoU potential.

#### 4. Better ASR transcripts
Replaced Whisper Tiny (WER ~30% on Russian) with faster-whisper large-v3-turbo (WER ~10%). Retranscribed all 436 audio files in parallel on 8x V100-32GB GPUs.

#### 5. Video path normalization in evaluation
`evaluate.py` now normalizes video paths (extracts `video_XXXX` ID) for comparison, preventing silent metric failures from path format mismatches.

## Architecture

```
OFFLINE (indexing):
  transcripts.pkl (faster-whisper large-v3-turbo)
    → Multi-scale chunker:
        sentence-level (3-10s each)     ← NEW: best for IoU
        group3 (3 consecutive segments) ← NEW
        short windows (20s/10s)         ← NEW
        medium windows (45s/15s)        ← NEW
        large windows (90s/30s)
        video summary
    → BGE-M3 embeddings (1024d, normalized)
    → numpy pickle index

ONLINE (<100ms per query):
  Query
    → BGE-M3 encode (~20ms)
    → Cosine similarity search top-50 (~1ms)
    → Dedup + deprioritize summary chunks
    → Adaptive padding per chunk scale
    → Return: 5 × (video_stem, start, end)
```

## File structure

```
attempt2/
├── config.py              # All parameters
├── ingest.py              # CLI: transcripts → chunks → embeddings → index
├── evaluate.py            # Local eval with Kaggle metrics (SR@K, VR@K)
├── submit.py              # Generate Kaggle submission CSV
├── pipeline/
│   ├── chunker.py         # Multi-scale chunking (5 levels + summary)
│   └── indexer.py         # BGE-M3 numpy index (build/load/search)
└── search/
    └── retriever.py       # Search with adaptive padding + reranker
```

## Usage

```bash
# 1. Ingest with new transcripts
python ingest.py --transcripts new_transcripts.pkl --index index_v2.pkl

# 2. Evaluate locally
python evaluate.py --sample 100

# 3. Generate submission
python submit.py --output submission_v2.csv
```

## IoU math explanation

Why multi-scale matters:

```
Ground truth:  |----15s----|
Attempt1 90s:  |--------------------90s--------------------|  IoU = 15/90 = 0.17 ✗
Sentence+pad:  |--8s--|---10s---|--8s--|                      IoU = 10/26 = 0.38 ~
Group3+pad:    |--6s--|------15s------|--6s--|                 IoU = 15/27 = 0.55 ✓
Short+pad:     |--4s--|--------20s--------|--4s--|            IoU = 15/28 = 0.53 ✓
```

The key insight: **smaller chunks with appropriate padding produce predicted windows closer to ground truth size, dramatically improving IoU.**

## Next improvements (TODO)
- [ ] HyDE (Hypothetical Document Embeddings) for query expansion
- [ ] Spellcheck via Yandex.Speller API for typo handling
- [ ] answer_en augmentation in index
- [ ] Sparse (BM25) retrieval via BGE-M3 lexical weights + RRF fusion
- [ ] SigLIP visual search as second retrieval channel
- [ ] Boundary refinement via cosine similarity hills
