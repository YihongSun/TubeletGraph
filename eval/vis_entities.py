import json, os, glob
import os.path as osp
import cv2
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pycocotools import mask as MaskUtils
import PIL.Image as Image
from tqdm import tqdm

import sys
sys.path.insert(0, osp.dirname(osp.dirname(__file__)))  # add proj dir to path
from utils import rle_to_bmask, bmask_to_rle, apply_anno, write_video, load_yaml_file, np_hvstack


def show_result(frame, obj_masks, only_obj_ind=None, overlay=True, num_colors=100):
    """ Show result by applying object masks to the frame.
    """
    if isinstance(frame, str):
        frame = np.array(Image.open(frame), dtype=np.uint8)
    elif isinstance(frame, np.ndarray):
        assert frame.dtype == np.uint8, "Expecting uint8 image"
    else:
        raise ValueError("Unsupported frame type")

    if overlay:
        vis_frame = frame.copy()
    else:
        vis_frame = np.zeros_like(frame)
    
    for obj_idx, mask in obj_masks.items():
        if only_obj_ind is not None and obj_idx not in only_obj_ind:
            continue
        if not type(mask) == np.ndarray:
            mask = rle_to_bmask(mask)
        
        if overlay:
            vis_frame = apply_anno(vis_frame, mask=mask, mask_color=int(obj_idx), num_colors=num_colors)
        else:
            
            vis_frame = apply_anno(vis_frame, mask=mask, mask_color=int(obj_idx), mask_alpha=1, num_colors=num_colors)
    return frame, vis_frame


def main(args):
    """ Main function to visualize entities.
    """
    ## load config
    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.dataset)

    ## load image paths
    video_dir = osp.join(data_cfg.image_dir, args.instance)
    frame_paths = sorted(glob.glob(osp.join(video_dir, data_cfg.image_format)))
    print('loaded frames:', len(frame_paths))

    ## load entity proposals
    supix_dirname = f'entities_{args.dataset}_{args.entity_method}'
    with open(osp.join(cfg.paths.intermdir, supix_dirname, args.instance+'.json'), 'r') as f:
        supix = json.load(f)
    assert len(supix) == len(frame_paths), f'Len of entities ({len(supix)}) != Len of image frames ({len(frame_paths)})'
    print('Loaded entity frames:', len(supix))

    ## visualize
    vis = [
        np_hvstack([
            [*show_result(frame, supix[str(frame_idx)])]
        ])
        for frame_idx, frame in enumerate(tqdm(frame_paths))
    ]

    ## save video
    out_path = osp.join(cfg.paths.visdir, 'entities', f'{supix_dirname}_{args.instance}.mp4')
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    write_video(out_path, vis, fps=data_cfg.fps)

    return out_path, vis


def get_parser():
    """ Get argument parser for visualization.
    """
    parser = argparse.ArgumentParser(description="Visualize all entities.")
    parser.add_argument('-c', "--config", default="configs/default.yaml", metavar="FILE", help="path to config file",)
    parser.add_argument('-d', '--dataset', type=str, default='vost', help='Dataset name')
    parser.add_argument('-m', '--entity_method', type=str, default='cropformer', help='entity proposal method')
    parser.add_argument('-i', '--instance', type=str, default='3161_peel_banana', help='instance name')

    return parser

if __name__ == "__main__":
    ## parse args
    args = get_parser().parse_args()
    out_path, vis = main(args)

    print('Saved entities visualization to', out_path)