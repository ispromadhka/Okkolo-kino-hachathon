from scenedetect import detect, ContentDetector
import hashlib

def detect_scenes(video_path: str, threshold: float = 27.0, min_scene_sec: float = 2.0) -> list[dict]:
    movie_id = hashlib.md5(video_path.encode()).hexdigest()[:12]
    scene_list = detect(video_path, ContentDetector(threshold=threshold))

    scenes = []
    for i, (start, end) in enumerate(scene_list):
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        duration = end_sec - start_sec
        if duration < min_scene_sec:
            continue
        scenes.append({
            "video_path": video_path,
            "movie_id": movie_id,
            "scene_index": i,
            "start_time": start_sec,
            "end_time": end_sec,
            "duration": duration,
        })
    return scenes
