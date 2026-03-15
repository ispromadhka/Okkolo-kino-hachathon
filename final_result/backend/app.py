"""
FastAPI backend for Video RAG Search.
Endpoints:
  GET /search?q=...&top_k=5  — search video fragments
  GET /video/{video_id}       — stream video file
  GET /health                 — health check
"""
import os
import re
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import pandas as pd
import csv
import time

app = FastAPI(title="Video RAG Search API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# === Globals (loaded once at startup) ===
MODEL = None
EMBEDDINGS = None
CHUNKS = None
TRAIN_Q_EMBS = None
TRAIN_ANSWERS = None
TRAIN_QIDS = None
GT_BY_QID = None
P75 = None
SEG_LOOKUP = None
VIDEO_DIR = None


@app.on_event("startup")
def load_models():
    global MODEL, EMBEDDINGS, CHUNKS, TRAIN_Q_EMBS, TRAIN_ANSWERS, TRAIN_QIDS
    global GT_BY_QID, P75, SEG_LOOKUP, VIDEO_DIR

    data_dir = os.environ.get("DATA_DIR", "data")
    model_path = os.environ.get("MODEL_PATH", "bge-m3-finetuned")
    transcripts_path = os.environ.get("TRANSCRIPTS_PATH", "new_transcripts.pkl")
    VIDEO_DIR = os.environ.get("VIDEO_DIR", os.path.join(data_dir, "video-rag", "videos"))

    print(f"Loading model from {model_path}...")
    MODEL = SentenceTransformer(model_path)

    # Load transcripts for segment lookup
    with open(transcripts_path, "rb") as f:
        transcripts = pickle.load(f)

    video_map = {}
    with open(os.path.join(data_dir, "video_files.csv")) as f:
        for row in csv.DictReader(f):
            m = re.search(r"(video_[a-f0-9]+)", row["video_path"])
            if m:
                video_map[m.group(1)] = row["video_path"]

    # Build segment lookup
    SEG_LOOKUP = {}
    for key, segs in transcripts.items():
        m = re.search(r"(video_[a-f0-9]+)", key)
        if m:
            vid = m.group(1)
            vfile = video_map.get(vid, f"videos/{vid}.mp4")
            SEG_LOOKUP[vfile] = segs

    # Build chunks
    from pipeline.chunker import merge_segments_to_window

    all_chunks = []
    for key, segs in transcripts.items():
        if not segs:
            continue
        m = re.search(r"(video_[a-f0-9]+)", key)
        vid = m.group(1) if m else key
        vfile = video_map.get(vid, f"videos/{vid}.mp4")
        for i, w in enumerate(merge_segments_to_window(segs, 90.0, 30.0)):
            all_chunks.append({
                "video_file": vfile, "start_time": w["start"], "end_time": w["end"],
                "text": w["text"], "chunk_index": i, "chunk_type": "window",
            })

    train = pd.read_csv(os.path.join(data_dir, "train_qa.csv"))
    P75 = (train["end"] - train["start"]).quantile(0.75)

    aug = 0
    for _, row in train.iterrows():
        answer = str(row.get("answer_en", "")).strip()
        if not answer or answer == "nan" or len(answer) < 20:
            continue
        if len(answer) > 1000:
            answer = answer[:1000]
        all_chunks.append({
            "video_file": row["video_file"],
            "start_time": float(row["start"]),
            "end_time": float(row["end"]),
            "text": answer,
            "chunk_index": 90000 + aug,
            "chunk_type": "answer_aug",
        })
        aug += 1

    CHUNKS = all_chunks
    EMBEDDINGS = MODEL.encode(
        [c["text"] for c in all_chunks],
        batch_size=32, show_progress_bar=True, normalize_embeddings=True,
    )

    # HyDE setup
    train_qs = train.drop_duplicates("question_id")
    TRAIN_Q_EMBS = MODEL.encode(train_qs["question_en"].tolist(), batch_size=32, normalize_embeddings=True)
    TRAIN_ANSWERS = train_qs["answer_en"].tolist()
    TRAIN_QIDS = train_qs["question_id"].tolist()
    GT_BY_QID = defaultdict(list)
    for _, row in train.iterrows():
        GT_BY_QID[int(row["question_id"])].append({
            "video_file": row["video_file"],
            "start": float(row["start"]),
            "end": float(row["end"]),
        })

    print(f"Loaded: {len(CHUNKS)} chunks, {EMBEDDINGS.shape} embeddings")


def get_transcript(video_file, start, end):
    """Get transcript text for a video segment."""
    segs = SEG_LOOKUP.get(video_file, [])
    texts = [s["text"] for s in segs if s["end"] > start and s["start"] < end and s["text"].strip()]
    return " ".join(texts) if texts else ""


@app.get("/search")
def search(q: str = Query(..., description="Search query"), top_k: int = 5):
    t0 = time.time()
    qv = MODEL.encode([q], normalize_embeddings=True)[0]

    sim = TRAIN_Q_EMBS @ qv
    bi = np.argmax(sim)
    bs = float(sim[bi])

    # Dynamic HyDE
    if bs > 0.6 and str(TRAIN_ANSWERS[bi]) != "nan":
        ae = MODEL.encode([str(TRAIN_ANSWERS[bi])[:500]], normalize_embeddings=True)[0]
        aw = np.clip(0.1 + (bs - 0.6) * (0.6 / 0.4), 0.1, 0.7)
        sv = (1 - aw) * qv + aw * ae
        sv = sv / np.linalg.norm(sv)
    else:
        top3 = np.argsort(EMBEDDINGS @ qv)[::-1][:3]
        c = EMBEDDINGS[top3].mean(0)
        c = c / np.linalg.norm(c)
        sv = 0.7 * qv + 0.3 * c
        sv = sv / np.linalg.norm(sv)

    scores = EMBEDDINGS @ sv
    top_idx = np.argsort(scores)[::-1][:20]

    seen = set()
    results = []
    for idx in top_idx:
        ch = CHUNKS[idx]
        key = (ch["video_file"], ch.get("chunk_index", 0))
        if key in seen:
            continue
        seen.add(key)

        # Adaptive boundary
        if ch.get("chunk_type") == "answer_aug":
            start = ch["start_time"]
            end = ch["end_time"]
        else:
            center = (ch["start_time"] + ch["end_time"]) / 2
            start = max(0, center - P75 / 2)
            end = center + P75 / 2

        video_id = re.search(r"(video_[a-f0-9]+)", ch["video_file"])
        video_id = video_id.group(1) if video_id else ch["video_file"]

        transcript = get_transcript(ch["video_file"], start, end)

        results.append({
            "rank": len(results) + 1,
            "video_id": video_id,
            "video_file": ch["video_file"],
            "start_time": round(start, 1),
            "end_time": round(end, 1),
            "score": round(float(scores[idx]), 4),
            "chunk_type": ch.get("chunk_type", ""),
            "transcript": transcript[:500],
        })
        if len(results) >= top_k:
            break

    latency = round((time.time() - t0) * 1000, 1)
    return {
        "query": q,
        "results": results,
        "latency_ms": latency,
        "hyde_used": bs > 0.6,
        "hyde_similarity": round(bs, 3),
    }


@app.get("/video/{video_id}")
def stream_video(video_id: str):
    """Stream video file by video_id (e.g. video_02578eb3)."""
    # Find video file on disk
    for ext in [".mp4", ".webm", ".mkv"]:
        path = Path(VIDEO_DIR) / f"{video_id}{ext}"
        if path.exists():
            media_type = {"mp4": "video/mp4", "webm": "video/webm", "mkv": "video/x-matroska"}
            return FileResponse(str(path), media_type=media_type.get(ext[1:], "video/mp4"))
    raise HTTPException(status_code=404, detail=f"Video {video_id} not found")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(CHUNKS) if CHUNKS else 0,
        "model": "bge-m3-finetuned",
    }
