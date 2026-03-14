"""
Generate Kaggle submission CSV.
Padding is applied per-chunk (adaptive, set during chunking/retrieval).
Usage: python submit.py [--index index_data.pkl] [--output submission.csv]
"""
import csv
import logging
import time
import argparse
from pathlib import Path
from tqdm import tqdm

from search.retriever import search

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def generate_submission(output_path='submission.csv'):
    t0 = time.time()

    queries = []
    with open('data/test.csv') as f:
        for row in csv.DictReader(f):
            queries.append({'query_id': row['query_id'], 'query': row['question']})
    log.info(f'{len(queries)} test queries')

    fieldnames = ['query_id']
    for i in range(1, 6):
        fieldnames.extend([f'video_file_{i}', f'start_{i}', f'end_{i}'])

    rows = []
    lats = []

    for q in tqdm(queries, desc='Generating'):
        qt = time.time()
        results = search(q['query'], top_k=5)
        lats.append(time.time() - qt)

        row = {'query_id': q['query_id']}
        for i in range(5):
            si = str(i + 1)
            if i < len(results):
                r = results[i]
                vname = Path(r['video_file']).stem
                row[f'video_file_{si}'] = vname
                row[f'start_{si}'] = round(r['start_time'], 1)
                row[f'end_{si}'] = round(r['end_time'], 1)
            else:
                row[f'video_file_{si}'] = ''
                row[f'start_{si}'] = 0.0
                row[f'end_{si}'] = 0.0
        rows.append(row)

    with open(output_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    avg_lat = sum(lats) / len(lats) if lats else 0
    log.info(f'Submission: {output_path} ({len(rows)} queries)')
    log.info(f'Avg latency: {avg_lat*1000:.0f}ms')
    log.info(f'Total: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='submission.csv')
    args = p.parse_args()
    generate_submission(args.output)
