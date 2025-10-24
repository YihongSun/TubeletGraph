# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import yaml
import random
# fmt: off
import sys
# fmt: on

import time, json
import os.path as osp
import cv2
import numpy as np
import tqdm
import torch

import pycocotools.mask as MaskUtils
sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(__file__))))  # add proj dir to path
from utils import load_yaml_file

def bmask_to_rle(binary_mask):
    assert binary_mask.dtype == bool, "Expecting binary mask"
    assert binary_mask.ndim == 2, "Expecting 2D mask"

    rle = MaskUtils.encode(np.asfortranarray(binary_mask))
    return {'counts': rle['counts'].decode('ascii'),
            'size': rle['size']}

def setup_cfg(config_file, opts):
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Runnign CropFormer to obtain initial region proposals.")
    parser.add_argument("-c", "--config", default="configs/default.yaml", metavar="FILE", help="path to config file",)
    parser.add_argument("-d", '--dataset', type=str, help='Dataset to run', default='vost')
    parser.add_argument("-s", '--split', type=str, default='val', help='list of img dirs to process')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers.')
    parser.add_argument('--wid', type=int, default=0, help='worker id.')
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.dataset)

    cf_cfg = cfg.entity_seg.cropformer
    sys.path[0] = osp.join(cf_cfg.project_path, 'demo_cropformer')
    sys.path.insert(1, cf_cfg.project_path)

    out_dir = osp.join(cfg.paths.intermdir, f'entities_{args.dataset}_cropformer')
    os.makedirs(out_dir, exist_ok=True)

    ### Loading after the sys.path modification
    from detectron2.data.detection_utils import read_image
    from predictor import VisualizationDemo

    confidence_threshold = cf_cfg.confidence_threshold
    cfg = setup_cfg(cf_cfg.config_path, cf_cfg.opts)
    demo = VisualizationDemo(cfg)
    
    with open(osp.join(data_cfg.split_dir, args.split+'.txt'), 'r') as f:
        vid_dirs = [line.strip() for line in f.readlines()]

        if args.num_workers > 1:    # shuffle if multiple workers
            random.seed(0); random.shuffle(vid_dirs)
            print('Shuffled:', ', '.join(vid_dirs[:args.num_workers]), '...')
            vid_dirs = vid_dirs[args.wid::args.num_workers]

    for vid_dir in vid_dirs:
        out_path = osp.join(out_dir, vid_dir + '.json')
        if os.path.exists(out_path):
            print(f"Skip {vid_dir} as {out_path} exists")
            continue
        frame_paths = glob.glob(os.path.join(data_cfg.image_dir, vid_dir, data_cfg.image_format))
        frame_paths = sorted(frame_paths)
        out = {}
        for path in tqdm.tqdm(frame_paths, desc=vid_dir):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions = demo.run_on_image(img)

            ##### color_mask
            pred_masks = predictions["instances"].pred_masks
            pred_scores = predictions["instances"].scores
            
            # select by confidence threshold
            selected_indexes = (pred_scores >= confidence_threshold)
            selected_scores = pred_scores[selected_indexes]
            selected_masks  = pred_masks[selected_indexes]
            _, m_H, m_W = selected_masks.shape
            mask_id = np.zeros((m_H, m_W), dtype=np.uint8)

            # rank
            selected_scores, ranks = torch.sort(selected_scores)
            ranks = ranks + 1
            for index in ranks:
                mask_id[(selected_masks[index-1]==1).cpu().numpy()] = int(index)
            unique_mask_id = np.unique(mask_id).tolist()
            unique_mask_id.pop(0)

            out[len(out)] = {ii: bmask_to_rle(mask_id==mid) for ii, mid in enumerate(unique_mask_id)}
        
        with open(out_path, 'w') as f:
            json.dump(out, f)