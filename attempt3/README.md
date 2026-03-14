# Attempt 3: Production code from cluster (score 0.500, 1st place)

This is the exact code that ran on the DGX-2 cluster and produced our best submission.

## Score: 0.500 (1st place on leaderboard)

## Pipeline

```
1. RETRANSCRIPTION (offline, 8x V100-32GB, ~40 min)
   Audio files (436) → faster-whisper large-v3-turbo (INT8)
   → retranscribe_parallel.py (8 GPU workers)
   → merge_transcripts.py → new_transcripts.pkl

2. INDEXING (offline, ~3 min)
   new_transcripts.pkl → 90s/30s sliding windows → 5010 chunks
   + train answer_en → 4466 answer augmentation chunks
   = 9476 total chunks → BGE-M3 1024d embeddings → index.pkl

3. SEARCH (online, <100ms per query)
   Query → BGE-M3 encode
   → HyDE: find similar train question, mix with answer embedding
   → Cosine similarity top-10 → dedup → ±10s padding
   → 5 × (video_stem, start, end)
```

## Key techniques that boosted score

| Technique | Score delta | Explanation |
|-----------|------------|-------------|
| Better ASR (Whisper Tiny → large-v3-turbo) | +0.01 | WER 30% → 10% |
| Answer augmentation | +0.09 | Train answers added as index chunks |
| HyDE query expansion | +0.03 | Mix query with similar train answer |
| **Total** | **+0.13** | **0.367 → 0.500** |

## Files

| File | Purpose |
|------|---------|
| `run_v12.py` | Complete v12 pipeline (the winning submission) |
| `retranscribe_parallel.py` | 8-GPU parallel ASR with faster-whisper |
| `retranscribe_fw.py` | Single-GPU ASR (slower fallback) |
| `merge_transcripts.py` | Merge per-GPU transcript pickles |
| `ingest.py` | Basic ingest (window chunks only, no answer aug) |
| `evaluate.py` | Local evaluation (SR@K, VR@K, FinalScore) |
| `submit.py` | Basic submission generator |
| `config.py` | Parameters |
| `pipeline/chunker.py` | Sliding window chunker |
| `pipeline/indexer.py` | BGE-M3 numpy index |
| `search/retriever.py` | Search with adaptive padding |

## Usage

```bash
# Symlink data
ln -sf /path/to/data data
ln -sf /path/to/new_transcripts.pkl new_transcripts.pkl

# Run the winning pipeline
python run_v12.py

# Output: submission_v12.csv
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

## Hardware used
- 8x Tesla V100-SXM3-32GB (DGX-2)
- 177 GB RAM
- Python 3.12

## What didn't work (experiments)
- Multi-scale chunking (171K chunks) → 0.289 (too noisy)
- Cascaded coarse+fine search → 0.328 (lost VR diversity)
- Removing summary chunks → 0.358 (summary helps VR)
- Segment refinement → 0.330 (over-narrowed windows)
- NVIDIA Parakeet TDT → CUDA errors on V100 (Volta not supported)
