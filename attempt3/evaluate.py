"""
Local evaluation on train data.
Computes SR@K, VR@K, FinalScore matching Kaggle metrics.
Usage: python evaluate.py [--padding 5.0] [--sample 100]
"""
import csv
import logging
import time
import argparse
from collections import defaultdict
from tqdm import tqdm

from search.retriever import search

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def compute_iou(pred_start, pred_end, gt_start, gt_end):
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0, inter_end - inter_start)
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    return intersection / union if union > 0 else 0


def load_train_data(path: str = 'data/train_qa.csv'):
    """Group ground truth by question_id."""
    gt_by_q = defaultdict(list)
    questions = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
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


def evaluate(padding: float = 5.0, sample: int = 0, use_ru: bool = False):
    t0 = time.time()
    gt_by_q, questions = load_train_data()
    qids = sorted(gt_by_q.keys())

    if sample > 0:
        import random
        random.seed(42)
        qids = random.sample(qids, min(sample, len(qids)))

    log.info(f'Evaluating on {len(qids)} questions (padding={padding}s)')

    all_preds = []
    all_gts = []

    for qid in tqdm(qids, desc='Evaluating'):
        q = questions[qid]
        query = q['question_ru'] if use_ru else q['question_en']
        results = search(query, top_k=5)

        preds = []
        for r in results:
            preds.append((
                r['video_file'],
                max(0, r['start_time'] - padding),
                r['end_time'] + padding,
            ))
        # Pad to 5 if needed
        while len(preds) < 5:
            if preds:
                preds.append(preds[-1])
            else:
                preds.append(('unknown', 0, 60))

        gts = [(g['video_file'], g['start'], g['end']) for g in gt_by_q[qid]]
        all_preds.append(preds)
        all_gts.append(gts)

    # Compute metrics
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
    final = (avg_sr + avg_vr) / 2

    log.info(f'  AvgSR={avg_sr:.4f}  AvgVR={avg_vr:.4f}')
    log.info(f'  FinalScore={final:.4f}')
    log.info(f'  Elapsed: {time.time()-t0:.1f}s')
    return final


def _sr_at_k(preds_list, gts_list, k):
    hits = 0
    for preds, gts in zip(preds_list, gts_list):
        for pv, ps, pe in preds[:k]:
            found = False
            for gv, gs, ge in gts:
                if pv == gv and compute_iou(ps, pe, gs, ge) >= 0.5:
                    found = True
                    break
            if found:
                hits += 1
                break
    return hits / len(preds_list)


def _vr_at_k(preds_list, gts_list, k):
    hits = 0
    for preds, gts in zip(preds_list, gts_list):
        gt_videos = {g[0] for g in gts}
        for pv, _, _ in preds[:k]:
            if pv in gt_videos:
                hits += 1
                break
    return hits / len(preds_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--padding', type=float, default=5.0)
    parser.add_argument('--sample', type=int, default=100, help='0=all, N=sample N questions')
    parser.add_argument('--ru', action='store_true', help='Use Russian queries')
    args = parser.parse_args()
    evaluate(padding=args.padding, sample=args.sample, use_ru=args.ru)
