"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
"""

import argparse, random, glob, os, sys
import multiprocessing as mp
import os.path as osp

# fmt: off
sys.path.insert(1, osp.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
import json
import pycocotools.mask as MaskUtils

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(__file__))))  # add proj dir to path
from utils import load_yaml_file, load_anno


def setup_cfg(fc_cfg):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_fcclip_config(cfg)
    cfg.merge_from_file(fc_cfg.config_path)
    cfg.merge_from_list(fc_cfg.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=['MODEL.WEIGHTS', 'checkpoints/fcclip_cocopan.pth'], nargs=argparse.REMAINDER,)
    return parser

def demo_run_on_image(demo, image, masks):
    ''' Digging into the function '''

    predictor = demo.predictor
    with torch.no_grad(): 
        if predictor.input_format == "RGB":
            image = image[:, :, ::-1]
        height, width = image.shape[:2]
        image = predictor.aug.get_transform(image).apply_image(image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image.to(predictor.cfg.MODEL.DEVICE)

        inputs = {"image": image, "height": height, "width": width, "masks": masks}
        predictions = predictor.model([inputs])[0]
        predictions['pooled_clip_feature'] = predictions['pooled_clip_feature'].cpu()

        return predictions

def get_parser():
    parser = argparse.ArgumentParser(description="Running FC-CLIP to obtain clip features.")
    parser.add_argument("-c", "--config", default="configs/default.yaml", metavar="FILE", help="path to config file",)
    parser.add_argument("-d", '--dataset', type=str, help='Dataset to run', default='vost')
    parser.add_argument("-s", '--split', type=str, default='val', help='list of img dirs to process')
    parser.add_argument("-t", '--tubelet_name', type=str, default='tubelets_vost_cropformer', help='tubelet directory name')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers.')
    parser.add_argument('--wid', type=int, default=0, help='worker id.')
    return parser
    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    my_cfg = load_yaml_file(args.config)
    data_cfg = getattr(my_cfg.datasets, args.dataset)
    fc_cfg = my_cfg.sem_sim
    assert fc_cfg.name == 'fcclip', f"Expect sem_sim.name to be 'fcclip' but got {fc_cfg.name}"

    sys.path[0] = osp.join(fc_cfg.project_path, 'demo')
    sys.path.insert(1, fc_cfg.project_path)
    from fcclip import add_maskformer2_config, add_fcclip_config
    from predictor import VisualizationDemo
    cfg = setup_cfg(fc_cfg)
    demo = VisualizationDemo(cfg)

    # Custom 
    args.tubelet_dir = osp.join(my_cfg.paths.intermdir, args.tubelet_name)
    out_dir = args.tubelet_dir.rstrip('/') + '_fcclip'
    os.makedirs(out_dir, exist_ok=True)

    with open(osp.join(data_cfg.split_dir, args.split+'.txt'), 'r') as f:
        split_names = [x.strip() for x in f.readlines()]

        instance_names = []
        for instance in split_names:
            init_prompt_path = sorted(glob.glob(osp.join(data_cfg.anno_dir, instance, data_cfg.anno_format)))[0]
            prompt_objs = load_anno(init_prompt_path)  # dict('1': rle, ...)
            instance_names += [(instance + '_' + k + '.json', instance) for k in prompt_objs.keys()]
        assert np.all([osp.exists(osp.join(args.tubelet_dir, x)) for x,y in instance_names])

        if args.num_workers > 1:    # shuffle if multiple workers
            random.seed(0); random.shuffle(instance_names)
            print('Shuffled:', ', '.join(instance_names[:args.num_workers]), '...')
            instance_names = instance_names[args.wid::args.num_workers]

    for anno_fname, video_name in instance_names:
        # if '555_tear_aluminium_foil' not in anno_fname:   # debug 555_tear_aluminium_foil 5304_unpack_broccoli 7359_fold_tape_measure
        #     continue

        out_path = osp.join(out_dir, anno_fname)
        if osp.exists(out_path):
            print(f"Skip {anno_fname} as {out_path} exists")
            continue

        # Loading annotations
        with open(osp.join(args.tubelet_dir, anno_fname), 'r') as f:
            load_data = json.load(f)
            all_tracks = load_data['all_tracks']
            tracked_objs = load_data['tracked_objs']
            init_tracked_objs = [obj_idx for obj_idx, obj_info in tracked_objs.items() if obj_info['init_frame_idx'] == 0]
            later_tracked_objs = [obj_idx for obj_idx, obj_info in tracked_objs.items() if obj_info['init_frame_idx'] > 0]
            prompt_obj = str(max([int(obj_idx) for obj_idx in init_tracked_objs]))
            candidate_objs = [obj_idx for obj_idx, obj_info in tracked_objs.items() if 'mm_iou' in obj_info and obj_info['mm_iou']>0]
            obj_ind_to_comp = set([prompt_obj] + candidate_objs)

        # fill in 0 to pad
        for obj_idx in later_tracked_objs:
            for metric_name in ['clip_sim', 'clip_sim_min', 'clip_sim_max', 'clip_sim_a', 'clip_sim_a_min', 'clip_sim_a_max']:
                tracked_objs[obj_idx][metric_name] = 0

        if len(candidate_objs) > 0:
            # Getting image paths
            frame_paths = glob.glob(osp.join(data_cfg.image_dir, video_name, data_cfg.image_format))
            frame_paths = sorted(frame_paths)

            out = dict()
            for ii, path in tqdm.tqdm(enumerate(frame_paths), desc=anno_fname.replace('.json', ''), total=len(frame_paths)):
                mdata = all_tracks[str(ii)]
                obj_ind_str = [k for k in mdata.keys() if k in obj_ind_to_comp]
                masks = np.stack([MaskUtils.decode([mdata[k]])[...,-1].astype(bool) for k in obj_ind_str])

                img = read_image(path, format="BGR")
                predictions = demo_run_on_image(demo, img, masks)

                mask_cnts = torch.sum(predictions['mask_for_pooling'] == 1, (2,3))
                out[str(ii)+'_cnts'] = {
                    k: mask_cnts[0, obj_ind_str.index(k)].item() for k in obj_ind_str
                }

                out[str(ii)] = {
                    k: predictions['pooled_clip_feature'][:,obj_ind_str.index(k)] for k in obj_ind_str if out[str(ii)+'_cnts'][k] > 0
                }

            frame_ind = list([int(x) for x in all_tracks.keys()]); frame_ind.remove(0); frame_ind.sort(); frame_ind = [str(x) for x in frame_ind]
            placeholder = torch.ones(1, 768) * torch.inf
            query_clip_feat = torch.cat([out[frame_idx][prompt_obj] if prompt_obj in out[frame_idx].keys() else placeholder for frame_idx in ['0'] + frame_ind])
            query_valid = query_clip_feat[:,0] < torch.inf
        
            query_clip_feat = F.normalize(query_clip_feat, dim=-1).T
            for obj_idx in candidate_objs:
                later_clip_feat_list = [out[frame_idx][obj_idx] for frame_idx in frame_ind if obj_idx in out[frame_idx].keys()]
                if len(later_clip_feat_list) > 0:
                    later_clip_feats = F.normalize(torch.cat(later_clip_feat_list), dim=-1)
                    cos_sim_all = later_clip_feats @ query_clip_feat[:, query_valid]

                    init_idx = tracked_objs[obj_idx]['init_frame_idx']
                    num_valid_prior = torch.sum(query_valid[:init_idx]).item()
                    cos_sim_prior = cos_sim_all[:, :num_valid_prior] if num_valid_prior > 0 else None
                else:
                    cos_sim_all, cos_sim_prior = None, None
                
                if cos_sim_prior is not None:
                    tracked_objs[obj_idx]['clip_sim'] = torch.mean(cos_sim_prior).item()
                    tracked_objs[obj_idx]['clip_sim_min'] = torch.min(cos_sim_prior).item()
                    tracked_objs[obj_idx]['clip_sim_max'] = torch.max(cos_sim_prior).item()
                if cos_sim_all is not None:
                    tracked_objs[obj_idx]['clip_sim_a'] = torch.mean(cos_sim_all).item()
                    tracked_objs[obj_idx]['clip_sim_a_min'] = torch.min(cos_sim_all).item()
                    tracked_objs[obj_idx]['clip_sim_a_max'] = torch.max(cos_sim_all).item()
        else:
            print(f"Skip {anno_fname} as no candidate objects found")

        with open(out_path, 'w') as f:
            json.dump({'tracked_objs': tracked_objs, 'all_tracks': all_tracks,}, f)