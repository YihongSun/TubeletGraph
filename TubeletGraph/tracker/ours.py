import numpy as np
import torch
from tqdm import tqdm

import sys, os, json
import os.path as osp
from pycocotools import mask as MaskUtils

class TubeletGraph():
    """ Tracking best superpixels
    """
    def __init__(self, tubelet_dir, thrds):
        self.tubelet_dir = tubelet_dir
        self.thrds = dict(thrds)
    
    def initialize(self, instance_name, video_dir):
        self.frame_paths = [osp.join(video_dir, f) for f in sorted(os.listdir(video_dir))]
        self.load_objs_compute_distances(instance_name)
        assert len(self.all_tracks) == len(self.frame_paths), f"Number of frames in {self.tubelet_dir} does not match video frames"

        if len(self.later_tracked_objs) > 0:
            metric_names = list(self.tracked_objs[self.later_tracked_objs[0]].keys())
            metric_names.remove('mask'); metric_names.remove('init_frame_idx')
            self.metrics = {key: {
                obj_idx: self.tracked_objs[obj_idx][key] for obj_idx in self.later_tracked_objs
            } for key in metric_names}
        else:
            self.metrics = None

    def load_objs_compute_distances(self, instance_name):
        all_track_path = osp.join(self.tubelet_dir, instance_name+'.json')
        print('Loading all tracks, clip feats, and multi-masks...')
        with open(all_track_path, 'r') as f:
            data = json.load(f)
            self.all_tracks = data['all_tracks']
            self.tracked_objs = data['tracked_objs']
            self.prompt_obj = str(np.max([int(obj_idx) for obj_idx, obj_info in self.tracked_objs.items() if obj_info['init_frame_idx'] == 0]))
            self.later_tracked_objs = [obj_idx for obj_idx, obj_info in self.tracked_objs.items() if obj_info['init_frame_idx'] > 0]

    def get_best_tracked_objs(self):
        if self.metrics is None:
            return [self.prompt_obj]

        keep = set(self.tracked_objs.keys())
        for metric_name, thrd in self.thrds.items():
            keep = keep & {obj_idx for obj_idx, metric in self.metrics[metric_name].items() if metric > thrd}
        print('Init obj idx:', self.prompt_obj, 'Added:', keep)

        return [self.prompt_obj]+list(keep)

    def track(self, instance_name, video_dir):
        
        def rle_wrapper(rle):
            return {'counts': rle['counts'].decode('ascii') if isinstance(rle['counts'], bytes) else rle['counts'],
            'size': rle['size']}
 
        self.initialize(instance_name, video_dir)

        num_frames = len(self.frame_paths)
        best_subset_indices = self.get_best_tracked_objs()
        output = {'prediction': dict(), 'supix_masks': dict()}

        for frame_idx in range(num_frames):
            mask_subset = {
                obj_idx: self.all_tracks[str(frame_idx)][str(obj_idx)]
                for obj_idx in best_subset_indices if str(obj_idx) in self.all_tracks[str(frame_idx)]
            }
            output['supix_masks'][frame_idx] = {
                best_subset_indices.index(obj_idx): rle_wrapper(mask) for obj_idx, mask in mask_subset.items()
            }
            output['prediction'][frame_idx] = {0: rle_wrapper(MaskUtils.merge(list(mask_subset.values()), intersect=0))}
    
        return output
    
    def clear_all_cache(self):
        pass