import json, os, glob, random, sys
import os.path as osp
import cv2
import argparse
import numpy as np
import torch
from types import SimpleNamespace
import matplotlib.pyplot as plt
from pycocotools import mask as MaskUtils
import PIL.Image as Image
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(__file__))))  # add proj dir to path
from TubeletGraph.tracker import get_tracker
from utils import rle_to_bmask, bmask_to_rle, coverage, load_yaml_file, save_all_to_json, load_anno


def propagate(sam2, all_tracks, start_frame, clean_prev_memory=20):
    obj_ids = [x for x in sam2.inference_state['obj_idx_to_id'].values()]
    if len(obj_ids) == 0:
        print('No objects to track')
        return
    print('Tracking {} objects from id={} to id={} [frame={} -> end]...'.format(len(obj_ids), min(obj_ids), max(obj_ids), start_frame))
    for out_frame_idx, out_obj_ids, out_mask_logits in sam2.predictor.propagate_in_video(sam2.inference_state):
        if out_frame_idx <= start_frame:
            continue

        if out_frame_idx not in all_tracks:
            all_tracks[out_frame_idx] = dict()
        for i, obj_idx in enumerate(out_obj_ids):
            all_tracks[out_frame_idx][obj_idx] = bmask_to_rle((out_mask_logits[i, 0] > 0.0).cpu().numpy())
        
        delete_f_idx = out_frame_idx - clean_prev_memory
        for obj_idx, obj_output in sam2.inference_state['output_dict_per_obj'].items():
            if delete_f_idx in obj_output['non_cond_frame_outputs'].keys():
                obj_output['non_cond_frame_outputs'][delete_f_idx].clear()
                deleted = True
        torch.cuda.empty_cache()

def propagate_fragment(sam2, all_tracks, start_frame, stop_frame):
    obj_ids = [x for x in sam2.inference_state['obj_idx_to_id'].values()]
    print('Tracking {} objects from id={} to id={} [frame={} -> frame={}]...'.format(len(obj_ids), min(obj_ids), max(obj_ids), start_frame, stop_frame))
    for out_frame_idx, out_obj_ids, out_mask_logits in sam2.predictor.propagate_in_video(sam2.inference_state):
        if out_frame_idx <= start_frame:
            continue

        if out_frame_idx not in all_tracks:
            all_tracks[out_frame_idx] = dict()
        for i, obj_idx in enumerate(out_obj_ids):
            all_tracks[out_frame_idx][obj_idx] = bmask_to_rle((out_mask_logits[i, 0] > 0.0).cpu().numpy())

        torch.cuda.empty_cache()
        if out_frame_idx >= stop_frame:
            break

def add_mask(sam2, obj_id, mask, all_tracks, tracked_objs, frame_idx):
    sam2.predictor.add_new_mask(
        inference_state=sam2.inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        mask=mask,
    )
    all_tracks[frame_idx][obj_id] = bmask_to_rle(mask)
    tracked_objs[obj_id] = {'mask': mask, 'init_frame_idx': frame_idx}

def find_entity_to_add(all_tracks, entity, start_frame, fill_coverage_thrd, pix_perc_thrd):
    """ 
    fill_coverage_thrd: For any new entity mask to add, the existing tracks must cover it < fill_coverage_thrd
    pix_perc_thrd: pixel percentage threshold for adding new object
    """    
    entity_masks_rle = entity[str(start_frame)]
    if len(entity_masks_rle) == 0:
        return []   # No super pixels found, skipped

    all_pred_masks = list(all_tracks[start_frame].values())
    img_h, img_w = next(iter(entity_masks_rle.values()))['size']
    if len(all_pred_masks) == 0:
        pred_cover_rle = bmask_to_rle(np.zeros((img_h, img_w), dtype=np.uint8) > 0)
    else:
        pred_cover_rle = MaskUtils.merge([x if type(x) is dict else bmask_to_rle(x) for x in all_pred_masks], intersect=0)

    coverages = np.array([coverage(den_mask=entity_mask_rle, mask2intersect=pred_cover_rle) for entity_idx, entity_mask_rle in entity_masks_rle.items()])
    pix_perc = np.array([MaskUtils.area(entity_mask_rle) for entity_idx, entity_mask_rle in entity_masks_rle.items()]) / img_h / img_w
    entity_to_add = np.where(np.logical_and(coverages < fill_coverage_thrd, pix_perc > pix_perc_thrd))[0]
    if len(entity_to_add) > 0:
        print('One side IOU:', ' '.join([f'{x:.2f}' for x in coverages[entity_to_add]]), '#Pixel(%):', ' '.join([f'{x:.2f}' for x in 100*pix_perc[entity_to_add]]))
    return entity_to_add

def setup(sam2, instance_name, vid_dir, entity_dir):
    entity_path = osp.join(entity_dir, instance_name+'.json')
    print('Loading from', entity_path)
    with open(entity_path, 'r') as f:
        entity = json.load(f)
    sam2.initialize(video_dir=osp.join(vid_dir, instance_name))
    num_frames = sam2.inference_state['num_frames']
    assert num_frames == len(entity.keys())
    return entity, num_frames, sam2

def compute_save_init_tubes(cfg, sam2, instance_name, save_path, max_obj_num=150):
    """ Compute and save initial tracks from super pixels """
    if osp.exists(save_path):
        return None
    entity, num_frames, sam2 = setup(sam2, instance_name, cfg.data.image_dir, cfg.entity_dir)        
    init_entity = entity[str(0)]
    tracked_objs = dict()
    all_tracks = {k: dict() for k in range(num_frames)}
    num_it = len(init_entity.items()) // max_obj_num + 1
    pix_thrd = entity['0']['0']['size'][0] * entity['0']['0']['size'][1] * cfg.init_entity_thrd
    
    for it in range(num_it):
        sam2.predictor.reset_state(sam2.inference_state)
        start = it * max_obj_num
        end = min(start+max_obj_num, len(init_entity.items()))
        for mask_idx, mask_rle in list(init_entity.items())[start:end]:
            if MaskUtils.area(mask_rle) < pix_thrd:
                print(f'Mask {mask_idx} is too small, skipped')
                continue
            add_mask(sam2, int(mask_idx), rle_to_bmask(mask_rle), all_tracks, tracked_objs, frame_idx=0)
        propagate(sam2, all_tracks, start_frame=0)

    dim = entity['0']['0']['size']
    save_all_to_json({'all_tracks': all_tracks, 'tracked_objs': tracked_objs}, save_path, dim=dim)
    sam2.predictor.reset_state(sam2.inference_state)
    return entity, num_frames, sam2


def resolve_init_entity_prompt_conflicts(cfg, sam2, all_tracks, tracked_objs, prompt_mask):
    """ Adapt the initial super pixels to the prompt mask """

    ### Remove small objects
    for obj_idx, obj in list(tracked_objs.items()):
        mask = obj['mask']  # RLE or np.ndarray
        mask_perc = MaskUtils.area(mask) / mask['size'][0] / mask['size'][1] if type(mask) is dict else np.mean(mask)
        if mask_perc < cfg.fill_entity_thrd:
            print(f'Object {obj_idx} ({mask_perc*100:.2f}%) is too small, removed.')
            tracked_objs.pop(obj_idx)
            for k, v in all_tracks.items():
                if obj_idx in v:
                    all_tracks[k].pop(obj_idx)
    
    ### Process overlaps between the prompt mask and the initial super pixels

    require_updates = dict()
    for obj_idx, obj in list(tracked_objs.items()):
        obj_coverage = float(coverage(den_mask=obj['mask'], mask2intersect=prompt_mask))
        if obj_coverage > cfg.fill_coverage_thrd:
            if obj_coverage < cfg.rm_init_entity_thrd:
                print(f'Object {obj_idx} has an overlap of {obj_coverage:.2f} with the prompt mask, modifying and retracking.')
                require_updates[obj_idx] = tracked_objs[obj_idx].copy()
            else:
                print(f'Object {obj_idx} has an overlap of {obj_coverage:.2f} with the prompt mask, removed.')

            tracked_objs.pop(obj_idx)
            for k, v in all_tracks.items():
                if obj_idx in v:
                    all_tracks[k].pop(obj_idx)
    
    if len(require_updates) > 0:
        sam2.predictor.reset_state(sam2.inference_state)
        for obj_idx, obj in require_updates.items():
            obj_mask_rle = obj['mask']
            obj_mask = rle_to_bmask(obj_mask_rle)
            intersect_mask = rle_to_bmask(MaskUtils.merge([obj_mask_rle, prompt_mask], intersect=1))
            obj_mask[intersect_mask] = 0
            add_mask(sam2, int(obj_idx), obj_mask, all_tracks, tracked_objs, frame_idx=0)
        propagate(sam2, all_tracks, start_frame=0)


def compute_pred(cfg, prompt_mask, instance_name, precomp_mm_path):
    """ Compute the precomputed masks.
    """
    sam2_mm = get_tracker(cfg.prox_tracker)
    sam2_mm.initialize(video_dir=osp.join(cfg.data.image_dir, instance_name))
    output = sam2_mm.track(mask=rle_to_bmask(prompt_mask), frame_idx=0)
    dim = prompt_mask['size']
    os.makedirs(osp.dirname(precomp_mm_path), exist_ok=True)
    save_all_to_json(output, precomp_mm_path, dim=dim)
    sam2_mm.clear_all_cache()
    torch.cuda.empty_cache()

def compute_save_all_tubes(cfg, sam2, instance_name, all_tracks, tracked_objs, prompt_mask, final_all_tracks_path):
    # Load precomputed tracks
    nothing_in_init = len(tracked_objs) == 0
    prompt_obj_id = max([int(x) for x in tracked_objs.keys()]) + 1 if not nothing_in_init else 1

    precomp_mm_path = osp.join(cfg.paths.intermdir, f'pred_{args.dataset}_{cfg.prox_tracker.name}', osp.basename(final_all_tracks_path))
    if not osp.exists(precomp_mm_path):
        compute_pred(cfg, prompt_mask, instance_name, precomp_mm_path)

    print(f'Adding Obj {prompt_obj_id} from {precomp_mm_path}')
    with open(precomp_mm_path, 'r') as f:
        data = json.load(f)
        multi_masks = {int(frame_idx): {int(kk): vv for kk, vv in v['0'].items()} for frame_idx, v in data['multi_masks'].items()}
        prediction = {int(k): v['0'] for k, v in data['prediction'].items()}
        for frame_idx in all_tracks.keys():
            if frame_idx in prediction:
                all_tracks[frame_idx][prompt_obj_id] = prediction[frame_idx]
    tracked_objs[prompt_obj_id] = {'mask': prediction[0], 'init_frame_idx': 0}

    print('Initial objects:', list(tracked_objs.keys()))
    print('Prompt object:', prompt_obj_id)

    dim = prompt_mask['size']
    if nothing_in_init: # turn off the tracker, nothing initially found to compete with
        save_all_to_json({'all_tracks': all_tracks, 'tracked_objs': tracked_objs}, final_all_tracks_path, dim=dim)
        return

    print('Running later object tracking...')
    num_frames = sam2.inference_state['num_frames']
    spacing = cfg.collect_spacing
    for collect_start_frame in range(spacing, num_frames+spacing, spacing):
        collection = set()
        scan_start = max(collect_start_frame - spacing, 1)
        scan_end = min(collect_start_frame, num_frames)
        for start_frame in range(scan_start, scan_end):
            sam2.predictor.reset_state(sam2.inference_state)

            entity_to_add = find_entity_to_add(all_tracks, entity, start_frame, cfg.fill_coverage_thrd, cfg.fill_entity_thrd)
            if len(entity_to_add) == 0:
                continue

            for entity_idx in entity_to_add:
                obj_id = max([int(x) for x in tracked_objs.keys()]) + 1
                obj_mask_rle = entity[str(start_frame)][str(entity_idx)]
                obj_mask = rle_to_bmask(obj_mask_rle)
                add_mask(sam2, obj_id, obj_mask, all_tracks, tracked_objs, frame_idx=start_frame)
                collection.add(obj_id)

                tracked_objs[obj_id]['mm_iou'] = float(np.max(MaskUtils.iou(
                    [multi_masks[start_frame][1], multi_masks[start_frame][2]], [obj_mask_rle], [False])
                ))
                tracked_objs[obj_id]['mm_cover'] = float(np.max([
                    coverage(obj_mask_rle, mask2intersect=multi_masks[start_frame][i]) for i in [1,2]
                ]))

            propagate_fragment(sam2, all_tracks, start_frame=start_frame, stop_frame=collect_start_frame)
        
        ## tracking the collection
        if collect_start_frame >= num_frames:
            print('Reached the end of the video, no need to track all, skipping.')
            continue
        if len(collection) == 0:
            print('No new objects found, skipping')
            continue

        sam2.predictor.reset_state(sam2.inference_state)
        for obj_id in collection:
            for reprompt_frame in range(collect_start_frame-6, collect_start_frame):
                if obj_id not in all_tracks[reprompt_frame]:
                    continue
                sam2.predictor.add_new_mask(
                    inference_state=sam2.inference_state,
                    frame_idx=reprompt_frame,
                    obj_id=obj_id,
                    mask=rle_to_bmask(all_tracks[reprompt_frame][obj_id]),
                )

        propagate(sam2, all_tracks, start_frame=collect_start_frame)
    
    save_all_to_json({'all_tracks': all_tracks, 'tracked_objs': tracked_objs}, final_all_tracks_path, dim=dim)


def make_config(args, cfg, verbose=True):
    tb_cfg = cfg.tubelet
    final_cfg = SimpleNamespace(
        tracker = tb_cfg.tracker,
        init_entity_thrd = tb_cfg.init_entity_thrd,
        fill_entity_thrd = tb_cfg.fill_entity_thrd,
        fill_coverage_thrd = tb_cfg.fill_coverage_thrd,
        rm_init_entity_thrd = tb_cfg.rm_init_entity_thrd,
        collect_spacing = int(tb_cfg.collect_spacing),
        prox_tracker = tb_cfg.prox_tracker,
        paths = cfg.paths,
        data = getattr(cfg.datasets, args.dataset),
        entity_method = tb_cfg.entity_method,
        entity_dir = osp.join(cfg.paths.intermdir, f'entities_{args.dataset}_{tb_cfg.entity_method}'),
        **args.__dict__,
    )
    if verbose:
        print("Config:")
        for arg in vars(final_cfg):
            print(f"  {arg.replace('_', ' ').title():15}: {getattr(final_cfg, arg)}")
    return final_cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Compute all tracks.")
    parser.add_argument('-c', "--config", default="configs/default.yaml", metavar="FILE", help="path to config file",)
    parser.add_argument('-d', '--dataset', type=str, default='vost', help='Dataset name')
    parser.add_argument('-s', '--split', type=str, help='Dataset split to run', default='val')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers.')
    parser.add_argument('--wid', type=int, default=0, help='worker id.')
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    yml_cfg = load_yaml_file(args.config)
    cfg = make_config(args, yml_cfg)

    assert cfg.init_entity_thrd <= cfg.fill_entity_thrd, 'should be <='
    assert cfg.prox_tracker.name == 'SAM2.1', 'Only SAM2.1 (multi mask) is supported for proximity similarity.'
    assert cfg.tracker.name == 'SAM2.1', 'Only SAM2.1 is supported for tubelet tracking.'

    # Load instance names to process
    with open(osp.join(cfg.data.split_dir, f'{cfg.split}.txt'), 'r') as f:
        instance_names = [x.strip() for x in f.readlines()]
        instance_names.sort()

        if cfg.num_workers > 1:    # shuffle if multiple workers
            random.seed(0); random.shuffle(instance_names)
            print('Shuffled:', ', '.join(instance_names[:cfg.num_workers]), '...')
            instance_names = instance_names[cfg.wid::cfg.num_workers]
    
    # Make model
    sam2 = get_tracker(cfg.tracker)

    # set save directories
    init_save_dir = osp.join(cfg.paths.intermdir, f'tubelets_{args.dataset}_{cfg.entity_method}_init')
    final_save_dir = init_save_dir.rstrip('_init')
    for d in [init_save_dir, final_save_dir]:
        if not osp.exists(d):
            os.makedirs(d)

    for instance_idx, instance_name in enumerate(instance_names):
        print('=' * 20)
        print(f"Processing ({instance_idx}/{len(instance_names)}) {instance_name}...")
        
        ### Step 1: Process all tracks from every super pixel in the initial frame, disregarding the mask prompt
        init_all_tracks_path = osp.join(init_save_dir, '{}.json'.format(instance_name))
        loaded = compute_save_init_tubes(cfg, sam2, instance_name, init_all_tracks_path)

        init_prompt_path = sorted(glob.glob(
            osp.join(cfg.data.anno_dir, instance_name, cfg.data.anno_format)
        ))[0]
        prompt_objs = load_anno(init_prompt_path)  # dict('1': rle, ...)

        for obj_idx, prompt_mask in prompt_objs.items():
            ### Process and all new tracks, using the mask prompt
            final_all_tracks_path = osp.join(final_save_dir, f'{instance_name}_{obj_idx}.json')
            if osp.exists(final_all_tracks_path):
                print('Final path found, skipped')
                continue
            
            with open(init_all_tracks_path, 'r') as f:
                data = json.load(f)
                all_tracks = {int(k): {int(k1): v1 for k1, v1 in v.items()} for k, v in data['all_tracks'].items()}
                tracked_objs = {int(k): v for k, v in data['tracked_objs'].items()}

            if loaded is None:
                entity, _, sam2 = setup(sam2, instance_name, cfg.data.image_dir, cfg.entity_dir)
            else:
                entity, _, sam2 = loaded
            
            resolve_init_entity_prompt_conflicts(cfg, sam2, all_tracks, tracked_objs, prompt_mask)
            compute_save_all_tubes(cfg, sam2, instance_name, all_tracks, tracked_objs, prompt_mask, final_all_tracks_path)
        
        ## clear cache after processing one video
        sam2.clear_all_cache()
        torch.cuda.empty_cache()
