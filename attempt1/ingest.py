"""Ingest: transcripts -> chunks -> embeddings -> numpy index."""
import pickle, logging, time, argparse, re
from tqdm import tqdm
from pipeline.chunker import build_chunks_for_video
from pipeline.indexer import build_index, get_embed_model
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

def run_ingest(window_sec=60.0, overlap_sec=15.0):
    t0 = time.time()
    video_map = {}
    with open('data/video_files.csv') as f:
        for row in csv.DictReader(f):
            m = re.search(r'(video_[a-f0-9]+)', row['video_path'])
            if m: video_map[m.group(1)] = row['video_path']
    log.info(f'Video map: {len(video_map)} entries')

    with open('data/transcripts.pkl','rb') as f:
        transcripts = pickle.load(f)
    log.info(f'Loaded {len(transcripts)} videos')

    all_chunks = []
    for key in tqdm(transcripts.keys(), desc='Chunking'):
        segs = transcripts[key]
        if not segs: continue
        m = re.search(r'(video_[a-f0-9]+)', key)
        vid = m.group(1) if m else key
        vfile = video_map.get(vid, f'videos/{vid}.mp4')
        chunks = build_chunks_for_video(vfile, segs, window_sec, overlap_sec)
        all_chunks.extend(chunks)

    log.info(f'Total chunks: {len(all_chunks)}')
    build_index(all_chunks)
    log.info(f'Ingest done in {time.time()-t0:.1f}s')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--window', type=float, default=60.0)
    p.add_argument('--overlap', type=float, default=15.0)
    a = p.parse_args()
    run_ingest(a.window, a.overlap)
