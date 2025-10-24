import json, cv2, os, sys
import os.path as osp
import numpy as np
from tqdm import tqdm
from PIL import Image
import imageio.v3 as imageio
import argparse
import ast
import base64, io
import openai
import time
from pycocotools import mask as MaskUtils

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(__file__))))  # add proj dir to path
from utils import rle_to_bmask, apply_anno, generate_rand_colors, load_yaml_file
from TubeletGraph.vlm.html_writer import HTMLWriter

def encode_image_from_np(image_np, is_rgb=True, resize=1):
    """ Convert a numpy array (HxWxC format) to base64 encoded image """
    assert image_np.ndim == 3 and image_np.shape[2] in [3, 4]
    
    if image_np.max() <= 1.0:
        image_np = (image_np * 255)   # 255 norm
    if not is_rgb:
        image_np = image_np[..., ::-1]  # Convert BGR to RGB
    
    image = Image.fromarray(image_np.astype(np.uint8))
    width, height = image.size
    if resize != 1.0:
        width = int(width * resize)
        height = int(height * resize)
        image = image.resize((width, height))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_string, 'image/png'

def get_image_payload(image, detail='low'):
    base64_image, mime_type = encode_image_from_np(image)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{base64_image}", "detail": detail},
    }

def get_text_payload(text):
    return {
        "type": "text",
        "text": text,
    }
    return payload

def get_user_content(data, html_writer=None, show_img_width = 600, image_detail='low'):
    if html_writer is not None:
        html_writer.add_heading('Prompt', level=3)
    content = list()
    for (data_type, x) in data:
        if data_type == 'text':
            content.append(get_text_payload(x))
            if html_writer is not None:
                html_writer.add_text(x)
        elif data_type == 'image':
            content.append(get_image_payload(x, image_detail))
            if html_writer is not None:
                html_writer.add_image(x)
        else:
            raise ValueError("Unsupported data type: {}".format(data_type))

    return content

def get_model_response(response, html_writer=None, sleep_time=0):
    time.sleep(sleep_time)  # to avoid rate limit
    rsp = response.choices[0].message.content
    if html_writer is not None:
        html_writer.add_heading('Response', level=3)
        html_writer.add_text(rsp)
    return rsp

def get_system_prompt():
    prompt = """
        You are a highly intelligent assistant that can analyze videos and images.
    """
    return prompt

def get_message_prompts():
    prompt_messages_id = [
        ('text', f"Here is an image, which is the first frame of a video. The object of interest is highlighted with a {init_c_name} contour."),
        ('text', f"Please name the object with {init_c_name} contour as concisely as possible in three words or less. Please do not include any other information in your answer."),
    ]
    prompt_messages_frame = [
        ('text', f"Do you able to recognize the objects in {init_c_name} and {query_c_name} contours clearly? Potential difficulties may be due to motion blur, or the majority of the objects are cropped by the edge of the image. Please answer with yes or no and do not include any other information."),
    ]
    prompt_messages_cls = [
        ('text', f"Here is an image, which is the first frame of a video. The first object of interest is highlighted with a {init_c_name} contour."),
        ('text', f"Here is another image, which is a later frame of the same video. A second object of interest is also highlighted with a {init_c_name} contour."),
        ('text', f"Are the two objects semantically the same? Please answer with yes or no and do not include any other information. Only say no if you can clearly recognize the object and know that they are semantically not related (e.g. a table and a chair)."),
    ]
    prompt_messages_action = [
        ('text', f"Here is an image, which is the first frame of a video. The object of interest is highlighted with a {init_c_name} contour."),
        ('text', f"Here is another image, which is a later frame of the same video. The original object of interest is still highlighted with a {init_c_name} contour. In addition, there is another object with a {query_c_name} contour that we believe is also a part of the original object of interest."),
        ('text', f"Please describe the two objects (one with {init_c_name} contour and one with {query_c_name} contour) in the second image each in three words or less while including the object name. In addition, please describe what is happening to the object of interest in the second image with a verb only without any tense. Please give the answer as a json tuple of (object with {init_c_name} contour, object with {query_c_name} contour, action). Please do not include any other information in your answer."),
    ]
    return prompt_messages_id, prompt_messages_frame, prompt_messages_cls, prompt_messages_action


def get_masked_image(image, mask, color_rgb, mask_alpha=0.0, contour_thickness=3):
    if isinstance(mask, dict):
        mask = rle_to_bmask(mask)
    masked_image = apply_anno(image.copy(), mask=mask, mask_color=color_rgb, mask_alpha=mask_alpha, contour_thickness=contour_thickness) # num_colors must be default
    return masked_image

def format_output(s):
    return s.replace('\n', '').replace('\"', '"').replace('\'', '"').replace('“', '"').replace('”', '"')

def get_added_track_starts(pred_data, prompt_obj_idx='0'):
    objs = dict()
    for frame_idx, frame_info in pred_data['supix_masks'].items():
        for obj_idx in frame_info :
            if obj_idx != prompt_obj_idx and obj_idx not in objs:
                objs[obj_idx] = frame_idx
    return objs, prompt_obj_idx

def get_parser():
    parser = argparse.ArgumentParser(description="Run object tracking methods.")
    parser.add_argument('-c', "--config", default="configs/default.yaml", metavar="FILE", help="path to config file",)
    parser.add_argument('-p', '--pred', type=str, help='prediction directory name', default='vost-val-Annotations_fps5-Ours')
    parser.add_argument('--temp', type=float, default=0.0, help='Temperature for sampling')
    return parser

def yes_no_cleanup(response):
    if 'not sure' in response.lower():
        return 'not sure'
    if 'no' in rsp.lower() and 'yes' not in rsp.lower():
        return 'no'
    if 'yes' in rsp.lower() and 'no' not in rsp.lower():
        return 'yes'
    return ''

def rle_wrapper(rle):
    return {'counts': rle['counts'].decode('ascii') if isinstance(rle['counts'], bytes) else rle['counts'],
    'size': rle['size']}

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.pred.split('-')[0])   # dataset config
    pred_track_dir = osp.join(cfg.paths.outdir, args.pred)

    model_name = cfg.vlm.model_name
    if model_name.startswith('Qwen'):
        client = openai.OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ.get("HF_API_KEY")
        )
    else:
        client = openai.OpenAI()

    init_color, init_c_name = np.array(cfg.vlm.init_color_rgb, dtype=float), cfg.vlm.init_color_name
    query_color, query_c_name = np.array(cfg.vlm.query_color_rgb, dtype=float), cfg.vlm.query_color_name
    out_dir = pred_track_dir + f'_' + model_name.replace('/', '_')
    os.makedirs(out_dir, exist_ok=True)
    
    system_prompt = get_system_prompt()
    prompt_messages_id, prompt_messages_frame, _, prompt_messages_action = get_message_prompts()

    instance_names = [x.removesuffix('.json') for x in os.listdir(pred_track_dir) if x.endswith('.json')]
    print('Override with all instances in annotation file.')
    instance_names = json.load(open('assets/vost_tas.json', 'r'))['instances']

    for instance_name in tqdm(instance_names, desc="Processing instances"):
        html_out_path = osp.join(out_dir, f'{instance_name}.html')
        new_pred_path = osp.join(out_dir, f'{instance_name}.json')

        if osp.exists(new_pred_path) and osp.exists(html_out_path):
            print(f"Skip {instance_name} as {new_pred_path} and {html_out_path} exists")
            continue

        html_writer = HTMLWriter(title=f"Prompt Results - {instance_name} - {cfg.vlm.model_name}")
        html_writer.add_heading('System-level Prompt', level=3)
        html_writer.add_text(system_prompt)

        # Load frame paths and predictions
        with open(osp.join(data_cfg.processed_anno_dir, instance_name + '.json'), 'r') as f:
            data = json.load(f)
            frame_paths = [osp.join(data['frame_dir'], f) for f in data['frame_filenames']]
        with open(osp.join(pred_track_dir, instance_name + '.json'), 'r') as f:
            pred_data = json.load(f)

        track_starts, prompt_obj_idx = get_added_track_starts(pred_data)
        
        to_remove = list()  # potentially remove objects that are not the same semantic class
        obj_info = dict()

        if len(track_starts) == 0:
            html_writer.add_text("No additional objects to track.")
        else:
            ## Stage 1: recognizing the object in the first frame
            init_img = imageio.imread(frame_paths[0])
            init_img_mask = get_masked_image(init_img, pred_data['prediction']['0'][prompt_obj_idx], init_color)

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": get_user_content([
                        prompt_messages_id[0],
                        ('image', init_img_mask),
                        prompt_messages_id[1],
                    ], html_writer=html_writer)}
                ],
                temperature=args.temp,
            )
            rsp = get_model_response(response, html_writer=html_writer)
            obj_info[prompt_obj_idx] = {'desc': rsp}

            ### Stage 2: Go through each object
            memory_responses = list()
            for obj_idx, first_obj_frame in track_starts.items():

                ## Stage 2.1: Figure out if this frame is good to analyze
                for i in range(10):
                    obj_start_frame = int(first_obj_frame) + i
                    if obj_start_frame >= len(frame_paths):
                        break
                    late_img = imageio.imread(frame_paths[obj_start_frame])

                    all_objs = list(pred_data['supix_masks'][str(obj_start_frame)].keys())
                    init_maks_union = MaskUtils.merge([pred_data['supix_masks'][str(obj_start_frame)][x] for x in all_objs if x != obj_idx], intersect=0)
                    new_track_mask = pred_data['supix_masks'][str(obj_start_frame)][obj_idx]
                    late_img_mask_ = get_masked_image(late_img, init_maks_union, init_color)
                    late_img_mask = get_masked_image(late_img_mask_, new_track_mask, query_color)

                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": get_user_content([
                                prompt_messages_frame[0],
                                ('image', late_img_mask),
                            ], html_writer=html_writer)}
                        ]
                    )
                    rsp = get_model_response(response, html_writer=html_writer)
                    if yes_no_cleanup(rsp) == 'yes':
                        break

                ## Stage 2.2: Figure out the two objects appears to originate from the same object - skipped
                # late_img_cls = late_img.copy()
                # late_img_cls[rle_to_bmask(init_maks_union)] = 255
                # late_img_mask_cls = get_masked_image(late_img_cls, new_track_mask, init_color)
                # response = client.chat.completions.create(
                #     model=model_name,
                #     messages=[
                #         {"role": "system", "content": system_prompt},
                #         {"role": "user", "content": get_user_content([
                #             prompt_messages_cls[0],
                #             ('image', init_img_mask),
                #             prompt_messages_cls[1],
                #             ('image', late_img_mask_cls),
                #             prompt_messages_cls[2],
                #         ], html_writer=html_writer)}
                #     ],
                #     temperature=args.temp,
                # )
                # rsp = get_model_response(response, html_writer=html_writer)
                # if yes_no_cleanup(rsp) == 'no':
                #     to_remove.append(obj_idx)
                #     continue
                
                ## Stage 2.3: Get the object part name and action

                contents = [
                    prompt_messages_action[0],
                    ('image', init_img_mask),
                    prompt_messages_action[1],
                    ('image', late_img_mask),
                    prompt_messages_action[2],
                ]
                for prev_frame_idx, prev_response in memory_responses:
                    t = (obj_start_frame - prev_frame_idx) // data_cfg.fps
                    contents.append(('text', f"Please take into considerations of your previous response for the same video at a frame {t} seconds ago:"))
                    contents.append(('text', prev_response))

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": get_user_content(contents, html_writer=html_writer)}
                    ],
                    temperature=args.temp,
                )
                rsp = get_model_response(response, html_writer=html_writer)
                memory_responses.append((obj_start_frame, rsp))
                parsed_rsp = ast.literal_eval([x for x in rsp.split('\n') if '```' not in x][0])
                obj_info[obj_idx] = {'desc': parsed_rsp[1], 'action': parsed_rsp[2], 'prior_desc': parsed_rsp[0], "analysis_frame_idx": obj_start_frame, "object_start_frame_idx": int(first_obj_frame)}

        if len(to_remove) > 0:
            to_remove = set(to_remove)
            for frame_idx in pred_data['prediction'].keys():
                new_supix_masks = {k: v for k, v in pred_data['supix_masks'][frame_idx].items() if k not in to_remove}
                pred_data['supix_masks'][frame_idx] = new_supix_masks
                pred_data['prediction'][frame_idx] = {"0": rle_wrapper(MaskUtils.merge(list(new_supix_masks.values()), intersect=0))}
        
        pred_data['obj_info'] = obj_info
        with open(new_pred_path, 'w') as f:
            json.dump(pred_data, f)
        html_writer.save(html_out_path)