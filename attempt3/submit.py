"""
Generate Kaggle submission from test queries.
Usage: python submit.py [--padding 5.0] [--output submission.csv]
"""
import csv
import logging
import time
import argparse
from tqdm import tqdm

from search.retriever import search

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def load_test_queries(path: str = 'data/test.csv') -> list[dict]:
    queries = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append({'query_id': row['query_id'], 'query': row['question']})
    log.info(f'Loaded {len(queries)} test queries')
    return queries


def generate_submission(output_path: str = 'submission.csv', padding: float = 5.0):
    t0 = time.time()
    queries = load_test_queries()

    fieldnames = ['query_id']
    for i in range(1, 6):
        fieldnames.extend([f'video_file_{i}', f'start_{i}', f'end_{i}'])

    rows = []
    latencies = []

    for q in tqdm(queries, desc='Searching'):
        qt0 = time.time()
        results = search(q['query'], top_k=5)
        lat = time.time() - qt0
        latencies.append(lat)

        row = {'query_id': q['query_id']}
        for i in range(5):
            idx = i + 1
            if i < len(results):
                r = results[i]
                start = max(0, r['start_time'] - padding)
                end = r['end_time'] + padding
                row[f'video_file_{idx}'] = r['video_file']
                row[f'start_{idx}'] = round(start, 2)
                row[f'end_{idx}'] = round(end, 2)
            else:
                last = results[-1] if results else None
                row[f'video_file_{idx}'] = last['video_file'] if last else 'videos/video_02578eb3.mp4'
                row[f'start_{idx}'] = 0.0
                row[f'end_{idx}'] = 60.0
        rows.append(row)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    log.info(f'Submission: {output_path} ({len(rows)} queries)')
    log.info(f'Latency: avg={avg_lat*1000:.0f}ms, p95={p95_lat*1000:.0f}ms')
    log.info(f'Total time: {elapsed:.1f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--padding', type=float, default=5.0)
    parser.add_argument('--output', type=str, default='submission.csv')
    args = parser.parse_args()
    generate_submission(output_path=args.output, padding=args.padding)
