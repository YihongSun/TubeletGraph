import json, os, sys, glob
import os.path as osp
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from pycocotools import mask as maskUtils
import multiprocessing as mp
from multiprocessing import Manager

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))  # add proj dir to path
from utils import load_yaml_file, rle_to_bmask, strip_instance_name, read_csv

def save_mask_png(mask, palette, out_path):
    mask = Image.fromarray(np.array(mask, dtype=np.uint8))
    mask.putpalette(palette)
    mask.save(out_path)

def json_to_png(args, cfg, data_cfg, png_dir):
    anno_dir=data_cfg.processed_anno_dir
    org_dir = osp.join(cfg.paths.outdir, args.pred)
    json_paths = glob.glob(osp.join(org_dir, '*.json'))
    json_paths.sort()

    out_annos = dict()
    for json_path in json_paths:

        with open(json_path.replace(org_dir, anno_dir), 'r') as f:
            data = json.load(f)
            instance_video_name = osp.basename(data['frame_dir'].rstrip('/'))
        
        if instance_video_name not in out_annos:
            # get obj_ind, palette, and size
            all_gt_paths = glob.glob(osp.join(data_cfg.anno_dir, instance_video_name, '*.png'))
            all_gt_paths.sort()
            mask = Image.open(all_gt_paths[0])
            palette = mask.getpalette()
            mask_np = np.array(mask)
            obj_ind = np.unique(mask_np)
            obj_ind = obj_ind[obj_ind != 0]; obj_ind = obj_ind[obj_ind != 255]
            obj_ind = obj_ind.tolist()
            # get the filenames
            filenames = data['frame_filenames']
            filenames = [x.replace('.jpg', '.png') for x in filenames]

            out_annos[instance_video_name] = {'filenames': filenames, 'palette': palette, 'size': mask_np.shape, 'obj_ind': obj_ind, 'predictions': {}}
        else:
            obj_ind = out_annos[instance_video_name]['obj_ind']
        
        obj_idx_str = osp.basename(json_path).replace('.json', '').replace(instance_video_name, '')
        if obj_idx_str.startswith('_'):
            obj_idx = int(obj_idx_str.lstrip('_'))
            assert obj_idx in obj_ind, f"Object index {obj_idx} not found in {instance_video_name}"
        elif obj_idx_str == '':
            assert len(obj_ind) == 1, f"Multiple objects found in the first frame: {instance_video_name}"
            obj_idx = int(obj_ind[0])
        else:
            raise ValueError(f"Invalid object index string: {obj_idx_str}")

        out_annos[instance_video_name]['predictions'][obj_idx] = json_path


    process_list = []
    sema = mp.Semaphore(12)
    manager = Manager()

    for instance_video_name, info in tqdm(out_annos.items(), desc=f'Converting {args.pred} to png'):

        def func():
            out_dir = osp.join(png_dir, instance_video_name)
            os.makedirs(out_dir, exist_ok=True)
            filenames = info['filenames']
            palette = info['palette']

            pred_to_write = [None for _ in range(len(filenames))]

            for obj_idx, json_path in info['predictions'].items():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    predictions = data['prediction']
                
                for frame_idx, pred in predictions.items():
                    bmask = rle_to_bmask(pred['0'])
                    if pred_to_write[int(frame_idx)] is None:
                        pred_to_write[int(frame_idx)] = bmask.astype(np.uint8) * obj_idx
                    else:
                        pred_to_write[int(frame_idx)][bmask] = obj_idx
            [
                save_mask_png(pred, palette, osp.join(out_dir, filenames[frame_idx])) for frame_idx, pred in enumerate(pred_to_write)
            ]

            sema.release()
            print(f"{instance_video_name} complete ! ")
        sema.acquire()
        p = mp.Process(target=func, args=())
        p.start()
        process_list.append(p)
    for process in tqdm(process_list):
        process.join()

def filter_lines_by_prefixes(input_file, prefixes,):
    out = dict()
    with open(input_file, 'r') as infile:
        for line in infile:
            for prefix in prefixes:
                if line.startswith(prefix):
                    out[prefix] = float(line.replace(prefix, '').strip())
                    break
    return out

def get_parser():
    parser = argparse.ArgumentParser(description="Convert the json outputs to png")
    parser.add_argument('-c', "--config", default="configs/default.yaml", metavar="FILE", help="path to config file",)
    parser.add_argument('-p', '--pred', type=str, help='prediction name to evaluate', required=True)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.pred.split('-')[0])

    ## compute pre/rec
    os.system(f"python3 eval/compute_pre_rec.py -c {args.config} -p {args.pred}")

    ## convert json to png and evaluate J scores
    png_dir = osp.join(cfg.paths.outdir, args.pred+'_png')
    json_to_png(args, cfg, data_cfg, png_dir)
    split = args.pred.split('-')[1]
    os.system(f"python3 thirdparty/RMem/evaluation/evaluation_method.py --results_path {png_dir} --dataset_path {data_cfg.data_dir} --set {split}")
    os.system(f"python3 eval/compute_split_score.py -c {args.config} -p {args.pred}")

    ## tally results
    tally = {}
    quant_txt = osp.join(cfg.paths.evaldir, args.pred+'.txt')
    pre_rec_data = filter_lines_by_prefixes(quant_txt, ['Precision:', 'Recall:', 'Precision(tr):', 'Recall(tr):'])
    tally.update({
        'P': pre_rec_data['Precision:'], 
        'R': pre_rec_data['Recall:'], 
        'P(tr)': pre_rec_data['Precision(tr):'], 
        'R(tr)': pre_rec_data['Recall(tr):']
    })
    for subsplit in ['', '_S', '_M', '_L']:
        quant_cvs = osp.join(png_dir, f'global_results-{split}{subsplit}.csv')
        data = {k:v for k,v in zip(*read_csv(quant_cvs))}
        tally.update({
            'J'+subsplit: 100*float(data['J-Mean']),
            'J(tr)'+subsplit: 100*float(data['J_last-Mean']),
        })
    
    
    tally_outpath = osp.join(cfg.paths.evaldir, f'{args.pred}_tally.txt')
    headers = ['J', 'J_S', 'J_M', 'J_L', 'P', 'R', 'J(tr)', 'J(tr)_S', 'J(tr)_M', 'J(tr)_L', 'P(tr)', 'R(tr)']
    header_str = ' &'.join(['      '] + headers)
    perf_str = '       &' + ' &'.join(['{:.1f}'.format(tally[h]) for h in headers])
    
    with open(tally_outpath, 'w') as f:
        f.write(header_str + '\n')
        f.write(perf_str + '\n')
    print(f"Results saved to {tally_outpath}")

