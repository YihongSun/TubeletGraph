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
from utils import rle_to_bmask, bmask_to_rle, apply_anno, write_video, load_yaml_file, np_hvstack, strip_instance_name


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
    """ Main function to visualize tubelets.
    """
    ## load config
    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.dataset)
    vid_instance = strip_instance_name(args.instance)
    obj_instance = args.instance

    ## load image paths
    video_dir = osp.join(data_cfg.image_dir, vid_instance)
    frame_paths = sorted(glob.glob(osp.join(video_dir, data_cfg.image_format)))
    print('loaded frames:', len(frame_paths))

    ## load entity proposals
    entity_dirname = f'entities_{args.dataset}_{args.entity_method}'
    with open(osp.join(cfg.paths.intermdir, entity_dirname, vid_instance+'.json'), 'r') as f:
        entity = json.load(f)
    assert len(entity) == len(frame_paths), f'Len of entities ({len(entity)}) != Len of image frames ({len(frame_paths)})'
    print('Loaded entity frames:', len(entity))

    ## load tubelets
    tubelet_dirname = f'tubelets_{args.dataset}_{args.entity_method}'
    if args.track_method != '':
        tubelet_dirname += f'_{args.track_method}'
    with open(osp.join(cfg.paths.intermdir, tubelet_dirname, obj_instance +'.json'), 'r') as f:
        data = json.load(f)
        all_tracks = data['all_tracks']
        tracked_objs = data['tracked_objs']
    init_objs = set([obj_idx for obj_idx, obj_info in tracked_objs.items() if obj_info['init_frame_idx'] == 0])
    later_objs = set([obj_idx for obj_idx, obj_info in tracked_objs.items() if obj_info['init_frame_idx'] > 0])
    print('init_objs:', len(init_objs))
    print('later_objs:', len(later_objs))

    ## visualize
    frames = [np.array(Image.open(frame), dtype=np.uint8) for frame in frame_paths]
    vis = [
        np_hvstack([
            [frame, show_result(frame, entity[str(frame_idx)])[1]],
            [show_result(frame, all_tracks[str(frame_idx)], init_objs)[1], show_result(frame, all_tracks[str(frame_idx)], later_objs)[1]]
        ])
        for frame_idx, frame in enumerate(tqdm(frames))
    ]

    ## save video
    out_path = osp.join(cfg.paths.visdir, 'tubelets', f'{tubelet_dirname}_{obj_instance}.mp4')
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    write_video(out_path, vis, fps=data_cfg.fps)

    return out_path, vis


def get_parser():
    """ Get argument parser for visualization.
    """
    parser = argparse.ArgumentParser(description="Visualize all tubelets.")
    parser.add_argument('-c', "--config", default="configs/default.yaml", metavar="FILE", help="path to config file",)
    parser.add_argument('-d', '--dataset', type=str, default='vost', help='Dataset name')
    parser.add_argument('-m', '--entity_method', type=str, default='cropformer', help='entity proposal method')
    parser.add_argument('-t', '--track_method', type=str, default='', help='entity propagate method (empty means sam2)')
    parser.add_argument('-i', '--instance', type=str, default='3161_peel_banana_1', help='instance name')

    return parser

if __name__ == "__main__":
    ## parse args
    args = get_parser().parse_args()
    out_path, vis = main(args)

    print('Saved tubelets visualization to', out_path)