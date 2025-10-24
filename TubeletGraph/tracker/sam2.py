import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

class SAM2:
    def __init__(self, 
                 model_weights = "_ckpts/sam2.1_hiera_large.pt", 
                 model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml", 
                 device = "cuda",
                 multi_mask=False):
        
        self.predictor = build_sam2_video_predictor(model_cfg, model_weights, device=device)
        self.multi_mask = multi_mask

    def initialize(self, **info):
        self.inference_state = self.predictor.init_state(video_path=info['video_dir'])
        
    def track(self, mask, frame_idx=0):
        self.predictor.reset_state(self.inference_state)
        _, _, _ = self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=0,
            mask=mask,
        )

        # output
        # |-> "prediction"
        #     |-> frame_idx
        #         |-> obj_id: np.ndarray
        # |-> "obj_score"
        #     |-> frame_idx
        #         |-> obj_id: float
        # |-> "multi_masks"
        #     |-> frame_idx
        #         |-> obj_id
        #             |-> rank: np.ndarray
        # |-> "multi_masks_pred_ious"
        #     |-> frame_idx
        #         |-> obj_id
        #             |-> rank: float

        output = {'prediction': dict(), 'obj_score': dict()}
        if self.multi_mask:
            output['multi_masks'] = dict()
            output['multi_masks_pred_ious'] = dict()

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            output['prediction'][out_frame_idx] = {
                obj_idx: (out_mask_logits[i, 0] > 0.0).cpu().numpy() 
                    for i, obj_idx in enumerate(out_obj_ids)
            }

            for i, obj_idx in enumerate(out_obj_ids):
                obj_out = self.inference_state['output_dict_per_obj'][obj_idx]
                if out_frame_idx not in obj_out['non_cond_frame_outputs']:
                    continue
                obj_frame_out = obj_out['non_cond_frame_outputs'][out_frame_idx]

                if out_frame_idx not in output['obj_score']:
                    output['obj_score'][out_frame_idx] = dict()
                output['obj_score'][out_frame_idx][obj_idx] = obj_frame_out['object_score_logits'][0,0].item()
                
                if self.multi_mask:
                    _, multimasks = self.predictor._get_orig_video_res_output(self.inference_state, obj_frame_out['low_res_multimasks'])
                    
                    if out_frame_idx not in output['multi_masks']:
                        output['multi_masks'][out_frame_idx] = dict()
                        output['multi_masks_pred_ious'][out_frame_idx] = dict()

                    ious_rank = np.argsort(obj_frame_out['ious'][i].cpu().numpy())[::-1]
                    output['multi_masks'][out_frame_idx][obj_idx] = {
                        ii: (multimasks[i, rank] > 0.0).cpu().numpy() 
                            for ii, rank in enumerate(ious_rank)
                    }
                    output['multi_masks_pred_ious'][out_frame_idx][obj_idx] = {
                        ii: obj_frame_out['ious'][i, rank].item() 
                            for ii, rank in enumerate(ious_rank)
                    }
        return output

    def clear_all_cache(self):
        for k in self.inference_state.keys():
            self.inference_state[k] = None
        torch.cuda.empty_cache()