"""
Chunking strategies for video transcripts.
Creates chunks at multiple granularity levels for optimal retrieval.
"""
import logging
from typing import Optional

log = logging.getLogger(__name__)


def merge_segments_to_window(segments: list[dict], window_sec: float = 60.0,
                              overlap_sec: float = 15.0) -> list[dict]:
    """
    Merge transcript segments into sliding windows.
    Each window ~ window_sec with overlap_sec overlap.
    Returns list of {start, end, text}.
    """
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

        # Advance by (window - overlap)
        step = window_sec - overlap_sec
        while i < len(segments) and segments[i]['start'] < window_start + step:
            i += 1
        if i == 0:
            i = 1  # safety: always advance

    return chunks


def create_video_summary_chunk(segments: list[dict], max_chars: int = 3000) -> dict:
    """
    Create one summary chunk for the entire video (all text concatenated).
    Used for VideoRecall@K (finding the right video).
    """
    full_text = ' '.join(s['text'].strip() for s in segments)
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]
    return {
        'start': segments[0]['start'] if segments else 0,
        'end': segments[-1]['end'] if segments else 0,
        'text': full_text,
    }


def build_chunks_for_video(video_file: str, segments: list[dict],
                           window_sec: float = 60.0,
                           overlap_sec: float = 15.0) -> list[dict]:
    """
    Build all chunk types for one video.
    Returns list of dicts ready for indexing.
    """
    chunks = []

    # 1. Sliding window chunks (main retrieval)
    windows = merge_segments_to_window(segments, window_sec, overlap_sec)
    for i, w in enumerate(windows):
        chunks.append({
            'video_file': video_file,
            'start_time': w['start'],
            'end_time': w['end'],
            'text': w['text'],
            'chunk_type': 'window',
            'chunk_index': i,
        })

    # 2. Video-level summary chunk (for VideoRecall)
    summary = create_video_summary_chunk(segments)
    chunks.append({
        'video_file': video_file,
        'start_time': summary['start'],
        'end_time': summary['end'],
        'text': summary['text'],
        'chunk_type': 'summary',
        'chunk_index': -1,
    })

    log.info(f'  {video_file}: {len(windows)} windows + 1 summary = {len(chunks)} chunks')
    return chunks
