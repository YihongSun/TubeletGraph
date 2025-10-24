import json, os, sys
import os.path as osp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from pycocotools import mask as maskUtils
sys.path.insert(0, osp.dirname(osp.dirname(__file__)))  # add proj dir to path
from utils import load_yaml_file

def compute_iou_rle(rlep, rleg, rle_ignore=None, val_when_both_empty=1.0):
    if rle_ignore is not None and maskUtils.area(rle_ignore) > 0:
        keep_mask = ~maskUtils.decode(rle_ignore)
        keep_rle = maskUtils.encode(np.asfortranarray(keep_mask.astype(np.uint8)))
        
        rlep = maskUtils.merge([rlep, keep_rle], intersect=True)
        rleg = maskUtils.merge([rleg, keep_rle], intersect=True)

    areap = maskUtils.area(rlep)
    areag = maskUtils.area(rleg)

    intersection = maskUtils.merge([rlep, rleg], intersect=True)
    intersection_area = maskUtils.area(intersection)

    union_area = areap + areag - intersection_area
    
    precision = intersection_area / areap if areap > 0 else None
    recall = intersection_area / areag if areag > 0 else None

    if union_area == 0:
        return val_when_both_empty, precision, recall
    
    return intersection_area / union_area, precision, recall

def get_performance(predictions2load, pred_dir, anno_dir, obj_id='0', skip_first=True, filter_inst=None, silent=False, compute_iou=True):
    anno_names = [f for f in os.listdir(anno_dir)]

    for pred_name in predictions2load:
        if pred_name == 'GT':
            continue
        anno_names = set([x for x in os.listdir(osp.join(pred_dir, pred_name)) if x in anno_names])
    
    if filter_inst is not None:
        anno_names = [f for f in anno_names if osp.splitext(f)[0] in filter_inst]
        
    anno_paths = [osp.join(anno_dir, f) for f in anno_names]

    m_str = ' / '.join(predictions2load)
    if not silent:
        print(f'Examples to evaluate for {m_str}: {len(anno_paths)}')
    if len(anno_paths) == 0:
        raise ValueError(f'No annotations found for {m_str}')

    masks = {p: {} for p in predictions2load}
    masks.update({'GT': {}, 'Ignore': {}})

    anno_frame_inds = dict()

    pbar = tqdm(anno_paths) if not silent else anno_paths
    for anno_path in pbar:
        # Load annotations
        with open(anno_path, 'r') as f:
            data = json.load(f)
            annotations = data['annotations']
            ignores = data['ignore']
            anno_frame_ind = [int(i) for i in annotations.keys()]
            if skip_first:
                anno_frame_ind.remove(0)
            anno_frame_ind.sort()

        masks['GT'][osp.basename(anno_path)] = [annotations[str(idx)] for idx in anno_frame_ind]
        masks['Ignore'][osp.basename(anno_path)] = [ignores[str(idx)] for idx in anno_frame_ind]

        for pred_name in predictions2load:
            pred_json = osp.join(pred_dir, pred_name, osp.basename(anno_path))
            with open(pred_json, 'r') as f:
                pred_data = json.load(f)
                predictions = pred_data['prediction']

            masks[pred_name][osp.basename(anno_path)] = [predictions[str(idx)][obj_id if idx != 0 else '0'] for idx in anno_frame_ind]
        
        anno_frame_inds[osp.basename(anno_path)] = anno_frame_ind

    ious, pres, recs = None, None, None
    if compute_iou:
        collection = {
            pred_name : {
                anno_name : [compute_iou_rle(pm, gm, im) for pm, gm, im in zip(masks[pred_name][anno_name], masks['GT'][anno_name], masks['Ignore'][anno_name])] 
                for anno_name in masks['GT'].keys()
            } 
            for pred_name in predictions2load
        }
        ious = {
            pred_name : {
                anno_name : [x[0] for x in collection[pred_name][anno_name]]
                for anno_name in masks['GT'].keys()
            } 
            for pred_name in predictions2load
        }
        pres = {
            pred_name : {
                anno_name : [x[1] for x in collection[pred_name][anno_name]]
                for anno_name in masks['GT'].keys()
            } 
            for pred_name in predictions2load
        }
        recs = {
            pred_name : {
                anno_name : [x[2] for x in collection[pred_name][anno_name]]
                for anno_name in masks['GT'].keys()
            } 
            for pred_name in predictions2load
        }
        
    return masks, ious, anno_frame_inds, pres, recs


def get_parser():
    parser = argparse.ArgumentParser(description="Run object tracking method with ground truth annotations.")
    parser.add_argument('-c', "--config", default="configs/default.yaml", metavar="FILE", help="path to config file",)
    parser.add_argument('-p', '--pred', type=str, help='prediction name to evaluate', required=True)
    return parser

def mean_wraper(x):
    non_none = [y for y in x if y is not None]
    return np.mean(non_none) if len(non_none) > 0 else None

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.pred.split('-')[0])
    masks, ious, anno_frame_inds, pres, recs = get_performance([args.pred], pred_dir=cfg.paths.outdir, anno_dir=data_cfg.processed_anno_dir, obj_id=cfg.eval.obj_id, skip_first=cfg.eval.skip_first_frame)

    ### Save results to txt file
    per_vid_meanIoU = {k: float(np.mean(v)) for k, v in ious[args.pred].items()}
    vid_names = list(per_vid_meanIoU.keys())
    vid_names.sort()
    
    avg_obj_size_prop = {fname : np.mean([maskUtils.area(mask) for mask in masks['GT'][fname]]) / masks['GT'][fname][0]['size'][0] / masks['GT'][fname][0]['size'][1] for fname in vid_names}
    pmin, pmax = 0, np.inf
    p33, p67 = np.percentile(list(avg_obj_size_prop.values()), 33), np.percentile(list(avg_obj_size_prop.values()), 67)
    scores = dict()
    for rng_name, rng in zip(['', '^S', '^M', '^L'], [(pmin, pmax), (pmin, p33), (p33, p67), (p67, pmax)]):
        vid_names_rng = [f for f in vid_names if rng[0] <= avg_obj_size_prop[f] < rng[1]]
        print(f'Range {rng_name}: {len(vid_names_rng)} videos from {rng[0]*100:.3f}% to {rng[1]*100:.3f}%')

        scores['meanIoU'+rng_name] = np.mean([np.mean(ious[args.pred][f]) for f in vid_names_rng])
        scores['Precision'+rng_name] = mean_wraper([mean_wraper(pres[args.pred][f]) for f in vid_names_rng])
        scores['Recall'+rng_name] = mean_wraper([mean_wraper(recs[args.pred][f]) for f in vid_names_rng])
        scores['acc25'+rng_name] = np.mean([np.mean([vi > 0.25 for vi in ious[args.pred][f]]) for f in vid_names_rng])
        scores['acc50'+rng_name] = np.mean([np.mean([vi > 0.50 for vi in ious[args.pred][f]]) for f in vid_names_rng])
        scores['acc75'+rng_name] = np.mean([np.mean([vi > 0.75 for vi in ious[args.pred][f]]) for f in vid_names_rng])

        last_quarter_ind = {f: int(np.floor(len(ious[args.pred][f]) * 0.75)) for f in vid_names_rng}
        scores['meanIoU_tr'+rng_name] = np.mean([np.mean(ious[args.pred][f][last_quarter_ind[f]:]) for f in vid_names_rng])
        scores['Precision_tr'+rng_name] = mean_wraper([mean_wraper(pres[args.pred][f][last_quarter_ind[f]:]) for f in vid_names_rng])
        scores['Recall_tr'+rng_name] = mean_wraper([mean_wraper(recs[args.pred][f][last_quarter_ind[f]:]) for f in vid_names_rng])
    
    output_txt_path = osp.join(cfg.paths.evaldir, f'{args.pred}.txt')

    with open(output_txt_path, 'w') as f:
        
        f.write(f'Prediction:          {args.pred}\n')
        f.write(f'Number of Instances: {len(vid_names)}\n')
        f.write('Mean IoU:            {}\n'.format(100*scores['meanIoU']))
        f.write('Mean IoU(tr):        {}\n'.format(100*scores['meanIoU_tr']))
        f.write('Precision:           {}\n'.format(100*scores['Precision']))
        f.write('Precision(tr):       {}\n'.format(100*scores['Precision_tr']))
        f.write('Recall:              {}\n'.format(100*scores['Recall']))
        f.write('Recall(tr):          {}\n'.format(100*scores['Recall_tr']))
        f.write('Acc@IoU=0.25:        {}\n'.format(100*scores['acc25']))
        f.write('Acc@IoU=0.50:        {}\n'.format(100*scores['acc50']))
        f.write('Acc@IoU=0.75:        {}\n'.format(100*scores['acc75']))
        f.write('Mean IoU^S:          {}\n'.format(100*scores['meanIoU^S']))
        f.write('Mean IoU(tr)^S:      {}\n'.format(100*scores['meanIoU_tr^S']))
        f.write('Acc@IoU=0.25^S:      {}\n'.format(100*scores['acc25^S']))
        f.write('Acc@IoU=0.50^S:      {}\n'.format(100*scores['acc50^S']))
        f.write('Acc@IoU=0.75^S:      {}\n'.format(100*scores['acc75^S']))
        f.write('Mean IoU^M:          {}\n'.format(100*scores['meanIoU^M']))
        f.write('Mean IoU(tr)^M:      {}\n'.format(100*scores['meanIoU_tr^M']))
        f.write('Acc@IoU=0.25^M:      {}\n'.format(100*scores['acc25^M']))
        f.write('Acc@IoU=0.50^M:      {}\n'.format(100*scores['acc50^M']))
        f.write('Acc@IoU=0.75^M:      {}\n'.format(100*scores['acc75^M']))
        f.write('Mean IoU^L:          {}\n'.format(100*scores['meanIoU^L']))
        f.write('Mean IoU(tr)^L:      {}\n'.format(100*scores['meanIoU_tr^L']))
        f.write('Acc@IoU=0.25^L:      {}\n'.format(100*scores['acc25^L']))
        f.write('Acc@IoU=0.50^L:      {}\n'.format(100*scores['acc50^L']))
        f.write('Acc@IoU=0.75^L:      {}\n'.format(100*scores['acc75^L']))
        f.write('\n')
        f.write(f'pred_dir: {cfg.paths.outdir}\n')
        f.write(f'anno_dir: {data_cfg.anno_dir}\n')
        f.write(f'object id: {cfg.eval.obj_id}\n')
        f.write(f'skip first frame: {cfg.eval.skip_first_frame}\n')
        f.write('\n')
        f.write(f'{"Video Name":25} {"Mean IoU":10}\n')
        f.write(f'{"-"*25} {"-"*10}\n')
        for video in vid_names:
            f.write(f'{osp.splitext(video)[0]:25} {100*per_vid_meanIoU[video]}\n')