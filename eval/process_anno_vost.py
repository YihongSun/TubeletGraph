import numpy as np
from pycocotools import mask as MaskUtils
import json
import os, sys
import os.path as osp
import numpy as np
from tqdm import tqdm
from PIL import Image
import mediapy as media
import cv2
import argparse

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))  # add proj dir to path
from utils import load_yaml_file, bmask_to_rle

def get_parser():
    parser = argparse.ArgumentParser(description="Run object tracking method with ground truth annotations.")
    parser.add_argument('-c', "--config", default="configs/default.yaml", metavar="FILE", help="path to config file",)
    parser.add_argument('-d', '--dataset', type=str, help='Dataset to run', default='vost', choices=['vost'])
    parser.add_argument('-s', '--split', type=str, help='Dataset split to run', default='val')
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)
    cfg.data = getattr(cfg.datasets, args.dataset)
    
    anno_dir = cfg.data.anno_dir
    img_dir = cfg.data.image_dir
    out_anno_dir = cfg.data.processed_anno_dir
    os.makedirs(out_anno_dir, exist_ok=True)

    with open(osp.join(cfg.data.split_dir, f'{args.split}.txt'), 'r') as file:
        vid_names = [x.strip() for x in file.readlines()]

    pbar = tqdm(vid_names)
    for vid_name in pbar:
        pbar.set_description(vid_name)

        original_anno_dir = osp.join(anno_dir, osp.splitext(vid_name)[0])
        frame_dir = osp.join(img_dir, osp.splitext(vid_name)[0])

        vid_anno = {
            'frame_dir': frame_dir,
        }
        
        org_filenames = os.listdir(frame_dir)
        org_filenames.sort()
        org_filenames

        vid_anno['frame_filenames'] = org_filenames

        init_anno_path = 'frame00000.png' if '5788_apply_paint' not in vid_name else 'frame00024.png'
        assert init_anno_path == osp.splitext(org_filenames[0])[0]+'.png'
        init_mask = np.array(Image.open(osp.join(original_anno_dir, init_anno_path)))
        unique_obj_ids = np.unique(init_mask)
        
        track_obj_ids = unique_obj_ids[np.logical_and(unique_obj_ids!=0,unique_obj_ids!=255)]
        loaded_anno = {int(a.split('.')[0].replace('frame', '')) : np.array(Image.open(osp.join(original_anno_dir, a))) for a in os.listdir(original_anno_dir)}
        vid_anno['ignore'] = {org_filenames.index(f'frame{k:05}.jpg') : bmask_to_rle(v == 255) for k, v in loaded_anno.items()}
        for obj_id in track_obj_ids:
            vid_anno['annotations'] = {org_filenames.index(f'frame{k:05}.jpg') : bmask_to_rle(v == obj_id) for k, v in loaded_anno.items()}

            with open(osp.join(out_anno_dir, f'{vid_name}_{obj_id}.json'), 'w') as f:
                json.dump(vid_anno, f)