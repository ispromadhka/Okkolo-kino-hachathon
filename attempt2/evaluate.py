"""
Local evaluation on train data with Kaggle-matching metrics.
Usage: python evaluate.py [--sample 100]
"""
import csv
import logging
import time
import argparse
import random
import re
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from search.retriever import search

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def compute_iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = (pe - ps) + (ge - gs) - inter
    return inter / union if union > 0 else 0


def load_train_data(path='data/train_qa.csv'):
    gt_by_q = defaultdict(list)
    questions = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            qid = int(row['question_id'])
            gt_by_q[qid].append({
                'video_file': row['video_file'],
                'start': float(row['start']),
                'end': float(row['end']),
            })
            if qid not in questions:
                questions[qid] = {
                    'question_en': row['question_en'],
                    'question_ru': row['question_ru'],
                }
    return gt_by_q, questions


def normalize_video(path):
    m = re.search(r'(video_[a-f0-9]+)', str(path))
    return m.group(1) if m else str(path)


def _sr_at_k(preds_list, gts_list, k):
    hits = 0
    for preds, gts in zip(preds_list, gts_list):
        for pv, ps, pe in preds[:k]:
            found = False
            for gv, gs, ge in gts:
                if normalize_video(pv) == normalize_video(gv) and compute_iou(ps, pe, gs, ge) >= 0.5:
                    found = True
                    break
            if found:
                hits += 1
                break
    return hits / len(preds_list)


def _vr_at_k(preds_list, gts_list, k):
    hits = 0
    for preds, gts in zip(preds_list, gts_list):
        gt_videos = {normalize_video(g[0]) for g in gts}
        for pv, _, _ in preds[:k]:
            if normalize_video(pv) in gt_videos:
                hits += 1
                break
    return hits / len(preds_list)


def evaluate(sample=100, use_ru=False):
    t0 = time.time()
    gt_by_q, questions = load_train_data()
    qids = sorted(gt_by_q.keys())

    if sample > 0:
        random.seed(42)
        qids = random.sample(qids, min(sample, len(qids)))

    log.info(f'Evaluating {len(qids)} questions')

    all_preds, all_gts = [], []
    for qid in tqdm(qids, desc='Evaluating'):
        q = questions[qid]
        query = q['question_ru'] if use_ru else q['question_en']
        results = search(query, top_k=5)

        preds = []
        for r in results:
            preds.append((r['video_file'], r['start_time'], r['end_time']))
        while len(preds) < 5:
            preds.append(preds[-1] if preds else ('x', 0, 60))

        gts = [(g['video_file'], g['start'], g['end']) for g in gt_by_q[qid]]
        all_preds.append(preds)
        all_gts.append(gts)

    for k in [1, 3, 5]:
        sr = _sr_at_k(all_preds, all_gts, k)
        vr = _vr_at_k(all_preds, all_gts, k)
        log.info(f'  SR@{k}={sr:.4f}  VR@{k}={vr:.4f}')

    sr1 = _sr_at_k(all_preds, all_gts, 1)
    sr3 = _sr_at_k(all_preds, all_gts, 3)
    sr5 = _sr_at_k(all_preds, all_gts, 5)
    vr1 = _vr_at_k(all_preds, all_gts, 1)
    vr3 = _vr_at_k(all_preds, all_gts, 3)
    vr5 = _vr_at_k(all_preds, all_gts, 5)

    avg_sr = (sr1 + sr3 + sr5) / 3
    avg_vr = (vr1 + vr3 + vr5) / 3
    fs = (avg_sr + avg_vr) / 2
    log.info(f'  AvgSR={avg_sr:.4f}  AvgVR={avg_vr:.4f}  FinalScore={fs:.4f}')
    log.info(f'  Elapsed: {time.time()-t0:.0f}s')
    return fs


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sample', type=int, default=100)
    p.add_argument('--ru', action='store_true')
    args = p.parse_args()
    evaluate(sample=args.sample, use_ru=args.ru)
