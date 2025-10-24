import json, os, sys, glob
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from pycocotools import mask as maskUtils

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))  # add proj dir to path
from utils import load_yaml_file, rle_to_bmask, read_csv

def get_parser():
    parser = argparse.ArgumentParser(description="Convert the json outputs to png")
    parser.add_argument('-c', "--config", default="configs/default.yaml", metavar="FILE", help="path to config file",)
    parser.add_argument('-p', '--pred', type=str, help='prediction name to evaluate', required=True)
    parser.add_argument('--subsplits', nargs='+', default=["val_S", "val_M", "val_L"], help='output directory')
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.pred.split('-')[0])
    png_dir = osp.join(cfg.paths.outdir, args.pred+'_png')
    args.split = args.pred.split('-')[1]

    per_instance_path = osp.join(png_dir, 'per-sequence_results-'+args.split+'.csv')
    data = read_csv(per_instance_path)
    per_instance_header, per_instance_perf = data[0], data[1:]
    print('Total instances:', len(per_instance_perf))
    sum = 0
    for subsplit in args.subsplits:
        split_path = osp.join(data_cfg.split_dir, f'{subsplit}.txt')
        with open(split_path, 'r') as f:
            split_instances = set([x.strip() for x in f.readlines()])
        
        split_per_instance_perf = [(float(J), float(Jtr)) for (k, J, Jtr) in per_instance_perf if k[:-2] in split_instances]
        sum += len(split_per_instance_perf)

        J_mean = np.mean([x[0] for x in split_per_instance_perf])
        Jtr_mean = np.mean([x[1] for x in split_per_instance_perf])
        print(f"Split {subsplit}: J-Mean: {J_mean:.3f}, Jtr-Mean: {Jtr_mean:.3f}")

        with open(osp.join(png_dir, f'global_results-{subsplit}.csv'), 'w') as f:
            f.write("J-Mean,J-Recall,J-Decay,J_last-Mean,J_last-Recall,J_last-Decay\n")
            f.write(f"{J_mean:.3f},0.000,,{Jtr_mean:.3f},0.000,")

    print(f"Total instances in {args.split} split: {sum} | total in {args.split} split: {len(per_instance_perf)}.")