import json
import os.path as osp
import argparse
import openai
from tqdm import tqdm
import numpy as np
import sys
from pycocotools import mask as MaskUtils
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))  # add proj dir to path
from utils import load_yaml_file

def judge_action_sem_acc(pred_sem, gt_sem, system_prompt, prompt_question):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model='gpt-4.1',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{
                "type": "text",
                "text": prompt_question.format(gt_sem.replace('_', ' '), pred_sem),
            }]}
        ],
        temperature=0.0,
    )
    rsp = response.choices[0].message.content.strip()
    return int(rsp)>0, rsp

def get_parser():
    parser = argparse.ArgumentParser(description="Run object tracking method with ground truth annotations.")
    parser.add_argument('-c', "--config", default="configs/default.yaml", metavar="FILE", help="path to config file",)
    parser.add_argument('-p', '--pred', type=str, help='prediction name to evaluate', required=True)
    parser.add_argument('-a', '--anno', type=str, default='assets/vost_tas.json', help='annotation path')
    return parser

def compute_action_acc(args, cfg):

    system_prompt = """
    You are a highly intelligent assistant that can analyze actions in text.
    """
    prompt_question = """
        Given a particular action description of “{0}”, is “{1}” similar to the verbs in this action? Please rate from -1 to 1, where -1 means completely unrelated, 0 means ambiguous, and 1 means “{1}” captures the meaning of “{0}” or is directly in it. Brief/general descriptions should still be considered as +1. Please answer with a single integer.'
    """
    
    pred_dir = osp.join(cfg.paths.outdir, args.pred)
    with open(args.anno, 'r') as f:
        data = json.load(f)
        instances = data['instances']       # list of instance names
        anno = data['annotations']          # dict of annotations:      instance_name -> anno dict
        frames = data['frames']             # dict of frame file paths: instance_name -> list of frame paths

    meta = ['instance[tf=id]:gt_time:pred_time:gt_action_desc:pred_action_desc:response:score']
    scores = []
    for instance in tqdm(instances):
        with open(osp.join(pred_dir, instance+'.json'), 'r') as f:
            data = json.load(f)
            obj_info = data['obj_info']

        tfms = anno[instance]['transformations']
        for tf_index, tf in enumerate(tfms):
            tf_start = tf['start_frame']
            tf_end = tf['end_frame']
            gt_action_desc = tf['action_desc']
            dist = len(frames[instance]); pred_action_desc = None
            for pred_obj in obj_info.values():
                if 'object_start_frame_idx' in pred_obj and tf_start <= pred_obj['object_start_frame_idx'] <= tf_end:
                    new_dist = abs(pred_obj['object_start_frame_idx'] - tf_end)
                    if new_dist < dist:
                        dist = new_dist
                        pred_action_desc = pred_obj['action']
            if pred_action_desc is not None:
                score, rsp = judge_action_sem_acc(pred_action_desc, gt_action_desc, system_prompt, prompt_question)
                scores.append(score)
                meta.append(f"{instance}[tf={tf_index}]:({tf_start},{tf_end}):{pred_obj['object_start_frame_idx']}:{gt_action_desc}:{pred_action_desc}:{rsp}:{score}")

    return scores, meta


def compute_object_acc(args, cfg):
    system_prompt = """
    You are a highly intelligent assistant that can analyze actions and resulting objects in text.
    """
    prompt_question = """
        Given the object description “{0}”, is “{1}” similar to it? Please rate from -1 to 1, where -1 means completely unrelated, 0 means ambiguous, and 1 means “{1}” is similar. Over- or under-specified descriptions should still be considered as +1. Please answer with a single integer.'
    """

    pred_dir = osp.join(cfg.paths.outdir, args.pred)
    with open(args.anno, 'r') as f:
        data = json.load(f)
        instances = data['instances']       # list of instance names
        anno = data['annotations']          # dict of annotations:      instance_name -> anno dict

    meta = ['instance[tf=id]:gt_object_desc:pred_object_desc:matched_ious:responses:scores']
    all_scores = []
    for instance in tqdm(instances):
        with open(osp.join(pred_dir, instance+'.json'), 'r') as f:
            data = json.load(f)
            pred_obj_info = data['obj_info']
            pred_obj_masks = data['supix_masks']

        tfms = anno[instance]['transformations']
        for tf_index, tf in enumerate(tfms):
            end_frame = tf['end_frame']
            pred_rles_to_match = pred_obj_masks[str(end_frame)]
            # gt_obj_desc, pred_obj_desc = [], []
            # for _, obj in tf['result_objects'].items():
            #     gt_rle = obj['mask']
            #     gt_obj_desc.append(obj['desc'])
                
            #     best_iou = 0
            #     best_pred_id = None
            #     for pred_id, pred_rle in pred_rles_to_match.items():
            #         iou = float(MaskUtils.iou([gt_rle], [pred_rle], [0])[0][0])
            #         if iou > best_iou:
            #             best_iou = iou
            #             best_pred_id = pred_id
            #     gt_matched_ious.append(best_iou)
            #     if best_iou >= 0.5 and best_pred_id in pred_obj_info:
            #         pred_obj_desc.append(pred_obj_info[best_pred_id]['desc'])
            #     else:
            #         pred_obj_desc.append(None)

            gt_objects = list(tf['result_objects'].values())
            gt_obj_desc = [obj['desc'] for obj in gt_objects]
            pred_obj_desc = [None]*len(gt_obj_desc)

            gt_rles = [obj['mask'] for obj in gt_objects]
            gt_matched_ious = [0]*len(gt_rles)
            pred_ids = list(pred_rles_to_match.keys())
            pred_rles = [pred_rles_to_match[pred_id] for pred_id in pred_ids]
            pred_matched_ious = [0]*len(pred_rles)
            
            # Compute IoU matrix: rows=GT, cols=predictions
            iou_matrix = np.zeros((len(gt_rles), len(pred_rles)))
            for i, gt_rle in enumerate(gt_rles):
                for j, pred_rle in enumerate(pred_rles):
                    iou_matrix[i, j] = float(MaskUtils.iou([gt_rle], [pred_rle], [0])[0][0])
            
            # Hungarian matching (maximize IoU = minimize negative IoU)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for i, j in zip(row_ind, col_ind):
                iou = float(iou_matrix[i, j])
                gt_matched_ious[i] = iou
                pred_matched_ious[j] = iou
                if iou >= 0.5 and pred_ids[j] in pred_obj_info:
                    pred_obj_desc[i] = pred_obj_info[pred_ids[j]]['desc']

            resps, scores = [], []
            for go, po in zip(gt_obj_desc, pred_obj_desc):
                if po is not None:
                    score, rsp = judge_action_sem_acc(po, go, system_prompt, prompt_question)
                else:
                    score, rsp = None, None
                scores.append(score)
                resps.append(rsp)
            meta.append(f"{instance}[tf={tf_index}]:{gt_obj_desc}:{pred_obj_desc}:{gt_matched_ious}:{pred_matched_ious}:{resps}:{scores}")
            all_scores += [s for s in scores if s is not None]
    return all_scores, meta

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)

    action_scores, action_meta = compute_action_acc(args, cfg)
    object_scores, object_meta = compute_object_acc(args, cfg)

    window_matched = {l.split(':')[0] for l in action_meta[1:]}
    action_sem_matched = {l.split(':')[0] for l in action_meta[1:] if l.split(':')[-1].strip() == 'True'}
    object_matched = {l.split(':')[0] for l in object_meta[1:] if np.all([iou > 0.5 for iou in eval(l.split(':')[-4])])}
    object_sem_matched = {l.split(':')[0] for l in object_meta[1:] if all([s == True for s in eval(l.split(':')[-1])])}

    sem_agn_match = window_matched & object_matched
    sem_match = action_sem_matched & object_sem_matched & object_matched

    num_transformations = len(object_meta) - 1
    sem_agn_acc = len(sem_agn_match) / num_transformations
    sem_acc = len(sem_match) / num_transformations

    print('Action Semantic Accuracy: {:.3f}'.format(np.mean(action_scores)))
    print('Object Semantic Accuracy: {:.3f}'.format(np.mean(object_scores)))
    print('Final Semantic Agnostic Accuracy: {:.3f}'.format(sem_agn_acc))
    print('Final Semantic Accuracy: {:.3f}'.format(sem_acc))

    with open(osp.join(cfg.paths.evaldir, f'sem_acc_{args.pred}.txt'), 'w') as f:
        f.write('Overall Action Semantic Accuracy: {:.3f}\n\n'.format(np.mean(action_scores)))
        f.write('Overall Object Semantic Accuracy: {:.3f}\n\n'.format(np.mean(object_scores)))
        f.write('Final Semantic Agnostic Accuracy: {:.3f}\n\n'.format(sem_agn_acc))
        f.write('Final Semantic Accuracy: {:.3f}\n\n'.format(sem_acc))

        f.write('--- Detailed Results ---\n')

        f.write('\n\nDetailed Final Semantic Agnostic results:\n')
        f.write('\n'.join(list(sem_agn_match)))

        f.write('\n\nDetailed Final Semantic results:\n')
        f.write('\n'.join(list(sem_match)))

        f.write('\n\nDetailed action results:\n')
        f.write('\n'.join(action_meta))

        f.write('\n\nDetailed object results:\n')
        f.write('\n'.join(object_meta))

