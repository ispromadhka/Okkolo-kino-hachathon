# Video RAG Search — Production Demo

## Score: 0.577 (1st place, Okko Hackathon)

## Quick Start

```bash
# 1. Place data files
ln -sf /path/to/data ./data
ln -sf /path/to/new_transcripts.pkl ./new_transcripts.pkl
ln -sf /path/to/bge-m3-finetuned ./bge-m3-finetuned

# 2. Launch
docker-compose up --build

# 3. Open browser
# Frontend: http://localhost:8501
# Backend API: http://localhost:8765/docs
```

## Without Docker

```bash
# Backend
cd backend
pip install -r requirements.txt
DATA_DIR=../data MODEL_PATH=../bge-m3-finetuned TRANSCRIPTS_PATH=../new_transcripts.pkl \
    VIDEO_DIR=../data/video-rag/videos uvicorn app:app --port 8765

# Frontend (separate terminal)
cd frontend
pip install -r requirements.txt
BACKEND_URL=http://localhost:8765 streamlit run app.py
```

## Architecture

```
Browser → Streamlit (port 8501)
              ↓
         FastAPI (port 8765)
              ↓
    Fine-tuned BGE-M3 + Qwen3-Embedding
    9476 chunks (numpy cosine search)
    Dynamic HyDE + Answer Augmentation
              ↓
    Video files (FileResponse streaming)
```

## API

```
GET /search?q=How+to+build+a+table&top_k=5
GET /video/video_02578eb3
GET /health
```

## Search response

```json
{
  "query": "How to build a table",
  "results": [
    {
      "rank": 1,
      "video_id": "video_abc123",
      "start_time": 45.2,
      "end_time": 139.2,
      "score": 0.8234,
      "chunk_type": "answer_aug",
      "transcript": "First you need to select the wood..."
    }
  ],
  "latency_ms": 82.3,
  "hyde_used": true,
  "hyde_similarity": 0.847
}
```

## Features
- Custom HTML5 video player with automatic start/end time
- Real-time transcript display for each fragment
- HyDE status indicator (shows when query expansion is active)
- Latency display (<100ms per query)
- Multilingual: Russian + English queries supported

## Scaling
- Backend is stateless (model loaded at startup)
- Can run N backend replicas behind nginx load balancer
- Index fits in RAM (~40MB for 9476 chunks)
- GPU optional (CPU inference ~200ms, GPU ~80ms)
