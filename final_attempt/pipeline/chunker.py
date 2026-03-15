import logging

log = logging.getLogger(__name__)


def merge_segments_to_window(segments, window_sec=60.0, overlap_sec=15.0):
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
            i = old_i + 1
    return chunks


def build_chunks_for_video(video_file, segments, window_sec=60.0, overlap_sec=15.0):
    chunks = []
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
    log.info(f'  {video_file}: {len(windows)} window chunks')
    return chunks
