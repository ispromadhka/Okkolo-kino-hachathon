from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
TEMP_DIR = PROJECT_ROOT / "temp"
KEYFRAMES_DIR = PROJECT_ROOT / "keyframes"

# === Models ===
GIGAAM_MODEL = "v3"
VLM_MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
VLM_DEVICE = "cuda:0"
VLM_MAX_TOKENS = 600

EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL_NAME = "Qwen/Qwen3-14B"
LLM_DEVICE = "cuda:1"

# === PySceneDetect ===
SCENE_THRESHOLD = 27.0
MIN_SCENE_LENGTH_SEC = 2.0

# === Keyframes ===
KEYFRAMES_PER_SCENE = 5

# === Qdrant ===
QDRANT_COLLECTION = "video_scenes"
QDRANT_IN_MEMORY = True

# === Search ===
SEARCH_TOP_K = 20
RERANK_TOP_K = 5

# === Post-processing ===
WINDOW_PADDING_SEC = 4.0  # expand predicted window by +-N sec for IoU

# === Enrichment ===
ENRICHMENT_CONTEXT_WINDOW = 2
