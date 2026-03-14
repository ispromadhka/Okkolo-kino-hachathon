"""
Multi-scale chunker with sentence-level, short, medium, and large windows.

Key insight: IoU >= 0.5 requires predicted window width close to ground truth.
- GT median = 59s, but range is 2s-1107s
- Sentence-level chunks (3-10s) with padding ±8s catch short GT fragments
- Short windows (20s) catch medium GT fragments
- Large windows (90s) catch long GT fragments + provide context for VR@K
"""
import logging

log = logging.getLogger(__name__)


def merge_segments_to_window(segments, window_sec=60.0, overlap_sec=15.0):
    """Sliding window over ASR segments. Snaps to segment boundaries."""
    if not segments:
        return []
    chunks = []
    i = 0
    while i < len(segments):
        window_start = segments[i]['start']
        window_end = window_start + window_sec
        texts = []
        j = i
        while j < len(segments) and segments[j]['start'] < window_end:
            texts.append(segments[j]['text'].strip())
            j += 1
        if texts:
            actual_end = segments[min(j, len(segments)) - 1]['end']
            chunks.append({
                'start': window_start,
                'end': actual_end,
                'text': ' '.join(texts),
            })
        old_i = i
        step = window_sec - overlap_sec
        while i < len(segments) and segments[i]['start'] < window_start + step:
            i += 1
        if i <= old_i:
            i = old_i + 1  # always advance
    return chunks


def build_chunks_for_video(video_file, segments, configs=None):
    """
    Build multi-scale chunks for one video.

    Returns list of dicts with:
      video_file, start_time, end_time, text, chunk_type, chunk_index, scale, padding
    """
    if configs is None:
        configs = [
            {"window": 20.0, "overlap": 10.0, "scale": "short", "padding": 4.0},
            {"window": 45.0, "overlap": 15.0, "scale": "medium", "padding": 2.0},
            {"window": 90.0, "overlap": 30.0, "scale": "large", "padding": 0.0},
        ]

    chunks = []
    idx_counter = 0

    # 1. Sentence-level chunks (highest IoU precision)
    for i, seg in enumerate(segments):
        if seg['text'].strip():
            chunks.append({
                'video_file': video_file,
                'start_time': seg['start'],
                'end_time': seg['end'],
                'text': seg['text'].strip(),
                'chunk_type': 'sentence',
                'chunk_index': idx_counter,
                'scale': 'sentence',
                'padding': 8.0,
            })
            idx_counter += 1

    # 2. Grouped sentence chunks (3 consecutive segments)
    for i in range(0, len(segments) - 2):
        group = segments[i:i+3]
        text = ' '.join(s['text'].strip() for s in group if s['text'].strip())
        if text:
            chunks.append({
                'video_file': video_file,
                'start_time': group[0]['start'],
                'end_time': group[-1]['end'],
                'text': text,
                'chunk_type': 'group3',
                'chunk_index': idx_counter,
                'scale': 'group3',
                'padding': 6.0,
            })
            idx_counter += 1

    # 3. Sliding window chunks at multiple scales
    for cfg in configs:
        windows = merge_segments_to_window(segments, cfg['window'], cfg['overlap'])
        for w in windows:
            chunks.append({
                'video_file': video_file,
                'start_time': w['start'],
                'end_time': w['end'],
                'text': w['text'],
                'chunk_type': 'window',
                'chunk_index': idx_counter,
                'scale': cfg['scale'],
                'padding': cfg['padding'],
            })
            idx_counter += 1

    # 4. Video-level summary (for VR@K)
    if segments:
        full_text = ' '.join(s['text'].strip() for s in segments if s['text'].strip())
        if len(full_text) > 3000:
            full_text = full_text[:3000]
        chunks.append({
            'video_file': video_file,
            'start_time': segments[0]['start'],
            'end_time': segments[-1]['end'],
            'text': full_text,
            'chunk_type': 'summary',
            'chunk_index': idx_counter,
            'scale': 'summary',
            'padding': 0.0,
        })

    return chunks
