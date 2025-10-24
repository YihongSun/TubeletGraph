import os, glob, sys
import os.path as osp
import argparse
import numpy as np
from types import SimpleNamespace

from tracker import get_tracker

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))  # add proj dir to path
from utils import load_yaml_file, load_anno, save_all_to_json

def make_config(args, cfg, verbose=True):
    new_cfg = SimpleNamespace(
        dataset = getattr(cfg.datasets, args.dataset),
        method = getattr(cfg.methods, args.method),
        split = args.split,
        outdir = my_cfg.paths.outdir,
        intermdir = my_cfg.paths.intermdir,
    )
    if verbose:
        print("Config:")
        for arg in vars(new_cfg):
            print(f"\t{arg.replace('_', ' ').title():10}: {getattr(new_cfg, arg)}")
    return new_cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Run object tracking methods.")
    parser.add_argument('-c', "--config", default="configs/default.yaml", metavar="FILE", help="path to config file",)
    parser.add_argument('-d', '--dataset', type=str, help='Dataset to run', default='vost', choices=['vost', 'm3vos', 'custom', 'davis', 'vscos'])
    parser.add_argument('-s', '--split', type=str, help='Dataset split to run', default='val')
    parser.add_argument('-m', '--method', type=str, help='Method to run', default='SAM2.1')
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    my_cfg = load_yaml_file(args.config)
    cfg = make_config(args, my_cfg)

    ## set tubelet dir
    cfg.method.tubelet_dir = osp.join(cfg.intermdir, cfg.method.tubelet_dirname)
    cfg.method.__dict__['_content'].pop('tubelet_dirname')

    ## gather instance names from split txt file
    instance_names = []
    with open(osp.join(cfg.dataset.split_dir, args.split+'.txt'), 'r') as f:
        split_names = [x.strip() for x in f.readlines()]

        for instance in split_names:
            init_prompt_path = sorted(glob.glob(osp.join(cfg.dataset.anno_dir, instance, cfg.dataset.anno_format)))[0]
            prompt_objs = load_anno(init_prompt_path)  # dict('1': rle, ...)
            instance_names += [(instance + '_' + k, instance, obj) for k, obj in prompt_objs.items()]
        assert np.all([osp.exists(osp.join(cfg.method.tubelet_dir, x+'.json')) for x,_,_ in instance_names])
    print(f"Running {cfg.method.name} on {len(instance_names)} annotations.\n")

    method = get_tracker(cfg.method)
    exp_name = '-'.join([cfg.dataset.name, cfg.split, cfg.method.name])
    out_dir = osp.join(cfg.outdir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    for instance, vid_name, obj in instance_names:
        save_path = osp.join(out_dir, instance + '.json')
        if osp.exists(save_path):
            print(f"Already computed: {save_path}")
            continue
        
        print(instance)
        pred = method.track(video_dir=osp.join(cfg.dataset.image_dir, vid_name), instance_name=instance)
        method.clear_all_cache()
        save_all_to_json(pred, save_path)