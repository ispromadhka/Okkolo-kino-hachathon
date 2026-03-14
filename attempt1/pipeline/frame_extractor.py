import subprocess
from pathlib import Path

def extract_keyframes(video_path: str, start_time: float, end_time: float,
                      scene_index: int, movie_id: str,
                      output_dir: str = "keyframes", num_frames: int = 5) -> list[str]:
    out = Path(output_dir) / movie_id / f"scene_{scene_index:04d}"
    out.mkdir(parents=True, exist_ok=True)
    duration = end_time - start_time
    paths = []
    for i in range(num_frames):
        frac = i / max(num_frames - 1, 1)
        ts = start_time + frac * duration
        p = out / f"frame_{i:02d}.jpg"
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
            "-vframes", "1", "-q:v", "2", str(p),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if p.exists():
            paths.append(str(p))
    return paths

def extract_audio(video_path: str, start_time: float, end_time: float,
                  output_path: str) -> bool:
    duration = end_time - start_time
    r = subprocess.run([
        "ffmpeg", "-y", "-ss", str(start_time), "-i", video_path,
        "-t", str(duration), "-ar", "16000", "-ac", "1", "-f", "wav", output_path,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0
