"""
Ingest pipeline: transcripts -> multi-scale chunks -> BGE-M3 embeddings -> index.
Usage: python ingest.py [--transcripts data/new_transcripts.pkl]
"""
import pickle
import logging
import time
import argparse
import csv
import re
from tqdm import tqdm

from pipeline.chunker import build_chunks_for_video
from pipeline.indexer import build_index, get_embed_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def run_ingest(transcripts_path='data/transcripts.pkl', index_path='index_data.pkl'):
    t0 = time.time()

    # Load video map
    video_map = {}
    with open('data/video_files.csv') as f:
        for row in csv.DictReader(f):
            m = re.search(r'(video_[a-f0-9]+)', row['video_path'])
            if m:
                video_map[m.group(1)] = row['video_path']
    log.info(f'Video map: {len(video_map)} entries')

    # Load transcripts
    with open(transcripts_path, 'rb') as f:
        transcripts = pickle.load(f)
    log.info(f'Loaded {len(transcripts)} videos from {transcripts_path}')

    # Build multi-scale chunks
    all_chunks = []
    for key, segs in tqdm(transcripts.items(), desc='Chunking'):
        if not segs:
            continue
        m = re.search(r'(video_[a-f0-9]+)', key)
        vid = m.group(1) if m else key
        vfile = video_map.get(vid, f'videos/{vid}.mp4')
        chunks = build_chunks_for_video(vfile, segs)
        all_chunks.extend(chunks)

    log.info(f'Total chunks: {len(all_chunks)}')

    # Count by scale
    scale_counts = {}
    for c in all_chunks:
        s = c.get('scale', 'unknown')
        scale_counts[s] = scale_counts.get(s, 0) + 1
    log.info(f'Chunks by scale: {scale_counts}')

    # Embed and save
    build_index(all_chunks, save_path=index_path)
    log.info(f'Ingest done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--transcripts', default='data/transcripts.pkl')
    p.add_argument('--index', default='index_data.pkl')
    args = p.parse_args()
    run_ingest(args.transcripts, args.index)
