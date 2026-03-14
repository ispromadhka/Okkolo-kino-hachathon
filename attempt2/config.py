from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

# === Models ===
EMBEDDING_MODEL = "BAAI/bge-m3"
ASR_MODEL = "large-v3-turbo"  # faster-whisper

# === Chunking (multi-scale) ===
CHUNK_CONFIGS = [
    {"window": 20.0, "overlap": 10.0, "scale": "short", "padding": 4.0},
    {"window": 45.0, "overlap": 15.0, "scale": "medium", "padding": 2.0},
    {"window": 90.0, "overlap": 30.0, "scale": "large", "padding": 0.0},
]
SENTENCE_PADDING = 8.0  # padding for sentence-level chunks

# === Search ===
SEARCH_TOP_K = 50
RERANK_TOP_K = 5

# === Index ===
INDEX_PATH = "index_data.pkl"
