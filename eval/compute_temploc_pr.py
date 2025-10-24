import json
import os.path as osp
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment

import sys
sys.path.insert(0, osp.dirname(osp.dirname(__file__)))  # add proj dir to path
from utils import load_yaml_file

def compute_pr(pred_time_stamps, gt_time_ranges, lpadding=1, rpadding=1):
    """
    Compute precision and recall with optimal bipartite matching using Hungarian algorithm.
    
    Args:
        pred_time_stamps: List of predicted timestamps
        gt_time_ranges: List of ground truth time ranges [(start, end), ...]
        lpadding: Left padding in minutes (converted to seconds)
        rpadding: Right padding in minutes (converted to seconds)
    
    Returns:
        tp, fp, fn, precision, recall
    """
    lpadding *= 30 # half of a second
    rpadding *= 30 # half of a second
    
    if len(pred_time_stamps) == 0:
        fn = len(gt_time_ranges)
        return 0, 0, fn, 1.0, 0.0 if fn > 0 else 1.0
    
    if len(gt_time_ranges) == 0:
        fp = len(pred_time_stamps)
        return 0, fp, 0, 0.0 if fp > 0 else 1.0, 1.0
    
    # Build cost matrix: rows = GTs, cols = preds
    # Cost = 0 if match (within padding), 1 otherwise
    # We use 1 instead of inf to avoid issues with linear_sum_assignment
    n_gt = len(gt_time_ranges)
    n_pred = len(pred_time_stamps)
    cost_matrix = np.ones((n_gt, n_pred))
    
    for i, gt in enumerate(gt_time_ranges):
        for j, pred in enumerate(pred_time_stamps):
            if pred >= gt[0] - lpadding and pred <= gt[1] + rpadding:
                cost_matrix[i, j] = 0  # Valid match
    
    # Use Hungarian algorithm to find optimal assignment
    # This minimizes total cost (maximizes valid matches)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Count true positives (valid matches with cost 0)
    tp = 0
    matched_preds = set()
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] == 0:  # Valid match
            tp += 1
            matched_preds.add(j)
    
    fn = n_gt - tp
    fp = n_pred - len(matched_preds)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    
    return tp, fp, fn, precision, recall

def test():
    pred_time_stamps = [180,180]
    gt_time_ranges = [(0, 120), (240, 360)]
    
    tp, fp, fn, precision, recall = compute_pr(
        pred_time_stamps, gt_time_ranges, lpadding=1, rpadding=1
    )
    
    print(f"Predictions: {pred_time_stamps}")
    print(f"Ground truths: {gt_time_ranges}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

def get_parser():
    parser = argparse.ArgumentParser(description="Run object tracking method with ground truth annotations.")
    parser.add_argument('-c', "--config", default="configs/default.yaml", metavar="FILE", help="path to config file",)
    parser.add_argument('-p', '--pred', type=str, help='prediction name to evaluate', required=True)
    parser.add_argument('-a', '--anno', type=str, default='assets/vost_tas.json', help='annotation path')
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.pred.split('-')[0])

    pred_dir = osp.join(cfg.paths.outdir, args.pred)
    with open(args.anno, 'r') as f:
        data = json.load(f)
        instances = data['instances']       # list of instance names
        anno = data['annotations']          # dict of annotations:      instance_name -> anno dict
        frames = data['frames']             # dict of frame file paths: instance_name -> list of frame paths
    
    EVAL_PAD = cfg.eval.temploc_max_pad + 1
    TP = {pad: 0 for pad in range(EVAL_PAD)}
    FP = {pad: 0 for pad in range(EVAL_PAD)}
    FN = {pad: 0 for pad in range(EVAL_PAD)}

    meta = ['padding, instance, gt_time, pred_time, tp, fp, fn']
    for pad in range(EVAL_PAD):
        for instance in instances:
            frame_names_int = [int(osp.splitext(osp.basename(x))[0].replace('frame', '')) for x in frames[instance]]

            with open(osp.join(pred_dir, instance+'.json'), 'r') as f:
                data = json.load(f)
                obj_info = data['obj_info']

            gt_time = [(frame_names_int[tf['start_frame']], frame_names_int[tf['end_frame']]) for tf in anno[instance]['transformations']]

            pred_time = [frame_names_int[y['object_start_frame_idx']] for x, y in obj_info.items() if x != '0']
            if anno[instance]['anno_end'] != -1: # annotation didn't end at the last frame
                pred_time = [pt for pt in pred_time if pt <= frame_names_int[anno[instance]['anno_end']]]

            tp, fp, fn, precision, recall = compute_pr(pred_time, gt_time, lpadding=pad, rpadding=pad)
            TP[pad] += tp
            FP[pad] += fp
            FN[pad] += fn

            meta.append(f"{pad}, {instance}, {gt_time}, {pred_time}, {tp}, {fp}, {fn}")

    precision = {k: TP[k] / (TP[k] + FP[k]) if (TP[k] + FP[k]) > 0 else 1 for k in TP.keys()}
    recall = {k: TP[k] / (TP[k] + FN[k]) if (TP[k] + FN[k]) > 0 else 1 for k in TP.keys()}
    for k in TP.keys():
        print(f"Pad {k}: pre={precision[k]:.3f}, rec={recall[k]:.3f}")
    
    with open(osp.join(cfg.paths.evaldir, f'temporal_loc_{args.pred}.txt'), 'w') as f:
        f.write('total padding (s), precision, recall\n')
        for k in TP.keys():
            f.write(f"{k}, {precision[k]:.3f}, {recall[k]:.3f}\n")
        f.write('\n\nDetailed results:\n')
        f.write('\n'.join(meta))


