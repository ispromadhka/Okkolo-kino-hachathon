import tempfile
from pathlib import Path
from pipeline.frame_extractor import extract_audio

_gigaam_model = None

def get_gigaam():
    global _gigaam_model
    if _gigaam_model is None:
        import gigaam
        _gigaam_model = gigaam.load_model("v2_ctc")
    return _gigaam_model

def transcribe_scene(video_path: str, start_time: float, end_time: float) -> str:
    model = get_gigaam()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        if not extract_audio(video_path, start_time, end_time, tmp.name):
            return ""
        if Path(tmp.name).stat().st_size < 1000:
            return ""
        result = model.transcribe(tmp.name)
    return result.strip() if isinstance(result, str) else str(result).strip()

def transcribe_full_video(video_path: str) -> str:
    """Transcribe entire video at once (for short 2-3 min videos)."""
    model = get_gigaam()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        if not extract_audio(video_path, 0, 999999, tmp.name):
            return ""
        result = model.transcribe(tmp.name)
    return result.strip() if isinstance(result, str) else str(result).strip()
