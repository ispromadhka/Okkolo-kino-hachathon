# Final Solution — Score 0.572 (1st Place)

## Competition
MultiLingual Video Fragment Retrieval Challenge (Okko Hackathon, March 14-15 2026)

Task: find exact video fragments by text query (RU/EN). Metric: FinalScore = (AvgSR + AvgVR) / 2.

## Score progression (37 experiments)

```
v4:  0.367 → Baseline (Whisper Tiny + BGE-M3 + 90s windows)
v8:  0.377 → Better ASR (faster-whisper large-v3-turbo)
v11: 0.470 → Answer augmentation (+0.093)
v12: 0.500 → HyDE query expansion (+0.030)
v15: 0.508 → P75 boundary truncation (+0.008)
v19: 0.518 → Dynamic HyDE + zero padding answers (+0.010)
v23: 0.567 → Fine-tuned BGE-M3, 3 epochs (+0.049)
v26: 0.572 → Fine-tuned BGE-M3, 4 epochs (+0.005) ← FINAL
```

## Architecture

```
OFFLINE (one-time):

  1. ASR: faster-whisper large-v3-turbo (8 GPU parallel)
     Audio files → new_transcripts.pkl (67K segments)

  2. Fine-tune: BGE-M3 on 1812 train question→chunk pairs
     4 epochs, MNR loss, batch=16, lr=2e-5 → bge-m3-finetuned/

  3. Index: 90s/30s sliding windows (5010) + answer_en augmentation (4466)
     = 9476 chunks → BGE-M3 embeddings (1024d) → numpy array

ONLINE (<100ms/query):

  Query → Fine-tuned BGE-M3 encode
  → Find similar train question (cosine)
  → IF sim > 0.95: return exact train GT timestamps (Direct GT)
  → IF sim > 0.6: Dynamic HyDE (mix query + answer embedding)
  → ELSE: Vector PRF (search → top-3 centroid → re-search)
  → Cosine search top-10 → dedup
  → Boundary: answer_aug → exact timestamps, windows → center ± P75/2
  → 5 × (video_stem, start, end)
```

## Key techniques and their impact

| Technique | Score delta | Why it works |
|-----------|-----------|--------------|
| Answer augmentation | +0.093 | Bridges vocabulary gap: queries use question-style, transcripts use lecture-style. Answers bridge both. |
| Fine-tuning BGE-M3 | +0.054 | Model learns domain-specific query→chunk mapping. 1812 pairs, 4 epochs = sweet spot. |
| HyDE query expansion | +0.030 | For queries similar to train, mixing answer embedding shifts search vector toward relevant content. |
| Dynamic HyDE weight | +0.010 | Variable weight (0.1-0.7) based on similarity. More confident match → more answer influence. |
| P75 boundary truncation | +0.008 | Windows (90s) are too wide for IoU. Center ± 47s matches typical GT fragment length. |
| Zero padding answers | +0.010 | Answer_aug chunks have exact GT timestamps. Adding padding only reduces IoU. |
| Better ASR | +0.010 | faster-whisper large-v3-turbo (WER ~10%) vs Whisper Tiny (WER ~30%). |

## What didn't work (lessons learned)

| Approach | Score | Lesson |
|----------|-------|--------|
| Multi-scale chunking (171K) | 0.289 | Too many small chunks flood results, kill VR@K |
| Hybrid sparse+dense | 0.404 | Sparse search disrupts ranking for this task |
| Cross-encoder reranker | 0.552 | Reranker promotes answer_aug from wrong videos |
| Doc2Query | 0.513 | Generated queries add noise to embeddings |
| Question augmentation | 0.477 | Similar questions create duplicate chunks |
| WiSE-FT weight interp | 0.545 | Dilutes strong fine-tuned signal |
| RRF fusion | 0.523 | Two search paths = inconsistent ranking |
| Semantic chunking | 0.563 | Shorter chunks = worse IoU even with expansion |
| More training data | 0.544 | Adding RU pairs + answers = overfitting |
| Multi-Bracket prediction | 0.484 | Using 3/5 slots for same video kills VR diversity |

## Files

```
final_attempt/
├── run.py                     # Complete pipeline (fine-tune + index + search)
├── evaluate.py                # Local evaluation (SR@K, VR@K, FinalScore)
├── retranscribe_parallel.py   # 8-GPU parallel faster-whisper retranscription
├── merge_transcripts.py       # Merge per-GPU transcript pickles
├── pipeline/
│   ├── chunker.py             # Sliding window chunker (90s/30s)
│   └── indexer.py             # BGE-M3 numpy index
└── search/
    └── retriever.py           # (optional, run.py is self-contained)
```

## Usage

```bash
# Prerequisites
pip install sentence-transformers faster-whisper numpy pandas tqdm static-ffmpeg

# Step 1: Retranscribe audio (8 GPU, ~40 min)
for GPU in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$GPU python retranscribe_parallel.py $GPU 8 &
done
wait
python merge_transcripts.py

# Step 2: Run pipeline (fine-tune + index + search, ~5 min)
ln -sf /path/to/data data
ln -sf /path/to/new_transcripts.pkl new_transcripts.pkl
WANDB_DISABLED=true python run.py

# Output: submission.csv
```

## Hardware
- 8x Tesla V100-SXM3-32GB (DGX-2)
- Fine-tuning: 78 seconds on 1 GPU
- Inference: <100ms per query
- Total pipeline: ~5 minutes (excluding ASR)

## Key insight

**Simplicity wins.** Every attempt to add complexity (rerankers, multi-scale, hybrid search, RRF, semantic chunking) made things worse. The winning formula is:
1. Good embeddings (fine-tuned BGE-M3)
2. Smart augmentation (answer_en from train)
3. Smart query expansion (dynamic HyDE)
4. Simple cosine search (numpy dot product)
5. Data-driven boundaries (P75 from train distribution)
