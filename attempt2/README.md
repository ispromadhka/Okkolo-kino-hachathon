# Attempt 2: Answer Augmentation + HyDE + Better ASR

## Score progression
| Version | Score | Key change |
|---------|-------|------------|
| v4 (attempt1) | 0.367 | 90s windows + Whisper Tiny + BGE-M3 |
| v8 | 0.377 | + faster-whisper large-v3-turbo (8 GPU parallel retranscription) |
| v9 | 0.289 | Multi-scale 171K chunks (too noisy, worse) |
| v10 | 0.328 | Cascaded coarse+fine (lost VR@K) |
| v11 | 0.470 | + answer_en augmentation from train data |
| **v12** | **0.500** | **+ HyDE query expansion (1st place)** |

## What works (v12 = 0.500, 1st place)

### 1. Better ASR transcripts
Replaced Whisper Tiny (WER ~30%) with faster-whisper large-v3-turbo (WER ~10%).
Retranscribed all 436 audio files in parallel on 8x V100-32GB in ~40 min.

### 2. Answer augmentation (+0.09 boost)
Train data has `answer_en` — detailed text answers for each question.
We add these answers as extra chunks in the index with the same (video_file, start, end).
This makes the index "denser" — queries match answer-style text, not just raw transcripts.

**Why it works:** Questions ask "How to do X?" but transcripts say "First you take the...".
The answer text bridges this semantic gap: "To do X, first you take the...".

### 3. HyDE query expansion (+0.03 boost)
For each test query:
1. Find the most similar train question (cosine similarity on BGE-M3)
2. If similarity > 0.7: mix query embedding with that train answer's embedding
3. Search with mixed embedding (0.6 * query + 0.4 * answer)

**Why it works:** The mixed embedding is semantically closer to the actual transcript
content than the raw question. It's a form of Hypothetical Document Embedding (HyDE).

### 4. Simple 90s window chunking (kept from v8)
Multi-scale chunking (v9, v10) made things worse by diluting results.
The simple 90s/30s sliding window with ±10s padding remains optimal.

## Architecture (v12)

```
OFFLINE:
  Audio files (436)
    → faster-whisper large-v3-turbo (8 GPU parallel)
    → new_transcripts.pkl (67K segments)

  new_transcripts.pkl
    → 90s/30s sliding window → 5010 window chunks
    → train answer_en → +4466 answer augmentation chunks
    → Total: 9476 chunks
    → BGE-M3 embeddings (1024d, normalized)
    → numpy pickle index

  train_qa.csv
    → BGE-M3 embeddings of question_en → train question index (for HyDE)

ONLINE (<100ms per query):
  Query
    → BGE-M3 encode
    → HyDE: find similar train question, mix with its answer embedding
    → Cosine similarity search top-10
    → Dedup by (video_file, chunk_index)
    → ±10s padding
    → Return: 5 × (video_stem, start, end)
```

## File structure

```
attempt2/
├── README.md
├── config.py                  # Parameters
├── run_v12.py                 # Complete v12 pipeline (score 0.500)
├── ingest.py                  # Basic ingest (without answer aug)
├── evaluate.py                # Local eval (SR@K, VR@K, FinalScore)
├── submit.py                  # Basic submission generator
├── retranscribe_parallel.py   # 8-GPU parallel retranscription
├── merge_transcripts.py       # Merge per-GPU transcript files
├── pipeline/
│   ├── chunker.py             # Sliding window + multi-scale chunking
│   └── indexer.py             # BGE-M3 numpy index
└── search/
    └── retriever.py           # Search with adaptive padding
```

## Usage

```bash
# Step 1: Retranscribe audio (8 GPU parallel, ~40 min)
for GPU in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$GPU python retranscribe_parallel.py $GPU 8 &
done
wait
python merge_transcripts.py

# Step 2: Run v12 pipeline (ingest + HyDE search + submission)
python run_v12.py

# Step 3: Submit
kaggle competitions submit ... -f submission_v12.csv
```

## What didn't work

| Approach | Score | Why it failed |
|----------|-------|---------------|
| Multi-scale 171K chunks (v9) | 0.289 | Sentence chunks flooded top-50 from same video, killed VR@K |
| Cascaded coarse+fine (v10) | 0.328 | Only 5 unique videos in top-5, less diversity |
| Removing summary chunks (v5) | 0.358 | Summary helped VR@K, removing it hurt |
| Refinement to best segments (v7) | 0.330 | Over-narrowed windows, missed ground truth |

## Key insight

**Don't over-engineer chunking.** Simple 90s windows + answer augmentation + HyDE
beats multi-scale, cascaded, and refinement approaches. The bottleneck is not
chunk granularity — it's semantic matching quality between queries and index.

## Next improvements (potential)
- [ ] Spellcheck via Yandex.Speller API for typo handling
- [ ] Sparse (BM25) retrieval + RRF fusion with dense
- [ ] Fine-tune BGE-M3 on train data (hard negative mining)
- [ ] SigLIP visual search as second retrieval channel
- [ ] Tune HyDE mixing weight (currently 0.6/0.4)
- [ ] Boundary refinement via cosine similarity hills
