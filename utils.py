import os
import os.path as osp
import cv2, colorsys, imageio, json
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from pycocotools import mask as MaskUtils

def rle_to_bmask(rle_mask):
    """
    Convert a RLE mask to a binary mask.
    """
    rle = [{
        'counts': rle_mask['counts'].encode('ascii') if isinstance(rle_mask['counts'], str) else rle_mask['counts'],
        'size': rle_mask['size']
    }]
    return MaskUtils.decode(rle)[...,-1].astype(bool)


def bmask_to_rle(binary_mask):
    """ Convert a binary mask to RLE mask.
    """
    assert binary_mask.dtype == bool, "Expecting binary mask"
    assert binary_mask.ndim == 2, "Expecting 2D mask"

    rle = MaskUtils.encode(np.asfortranarray(binary_mask))
    return {'counts': rle['counts'].decode('ascii'),
            'size': rle['size']}


def generate_rand_colors(num_colors, seed=0, lightness=1, saturation=1):
    """ Generate a list of random colors in RGB format.
    """
    uniform_colors = [colorsys.hsv_to_rgb(i / num_colors, saturation, lightness) for i in range(num_colors)]
    uniform_colors = np.array(uniform_colors) * 255

    np.random.seed(seed)
    np.random.shuffle(uniform_colors)

    return uniform_colors


def apply_anno(image, prompts=None, mask=None, mask_color=0, mask_alpha=0.3, contour_color=None, contour_thickness=2, num_colors=10):
    """ Apply annotations to an image.    
    """
    if prompts is not None:
        coords, labels = prompts['points'], prompts['labels']
        for point in coords[labels==1]:
            cv2.drawMarker(image, tuple(point.astype(int)), color=(255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=17, thickness=5, line_type=cv2.LINE_AA)
            cv2.drawMarker(image, tuple(point.astype(int)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2, line_type=cv2.LINE_AA)
        for point in coords[labels==0]:
            cv2.drawMarker(image, tuple(point.astype(int)), color=(255, 255, 255), markerType=cv2.MARKER_STAR, markerSize=17, thickness=5, line_type=cv2.LINE_AA)
            cv2.drawMarker(image, tuple(point.astype(int)), color=(0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=15, thickness=2, line_type=cv2.LINE_AA)

    if mask is not None:
        if type(mask_color) == int:
            mask_color = generate_rand_colors(num_colors, lightness=1, seed=10)[mask_color % num_colors]
        
        if contour_color is None:
            contour_color = mask_color
        elif type(contour_color) == int:
            contour_color = generate_rand_colors(num_colors, lightness=1, seed=10)[contour_color % num_colors]
        
        image[mask] = (image[mask].astype(float) * (1 - mask_alpha) + mask_alpha * mask_color).astype(np.uint8)
        if contour_thickness > 0:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color=(contour_color[0], contour_color[1], contour_color[2]), thickness=contour_thickness, lineType=cv2.LINE_AA)

    return image


def apply_func_to_leaves(data, function=None):
    """ Recursively applies a function to all leaf values in a nested data structure.
    """

    if isinstance(data, dict):
        return {k: apply_func_to_leaves(v, function) for k, v in data.items()}
    elif isinstance(data, list):
        return [apply_func_to_leaves(v, function) for v in data]
    elif isinstance(data, tuple):
        return tuple(apply_func_to_leaves(v, function) for v in data)
    elif isinstance(data, set):
        # Note: sets can only contain hashable elements, so they may not work for all nested structures
        return {apply_func_to_leaves(v, function) for v in data}
    else:
        # This is a leaf (non-container value) - transform it if a function is provided
        return function(data) if function is not None else data


def save_all_to_json(prediction, path, dim=None):
    """ Convert all binary 2d masks in a prediction to RLE format and save prediction to json path.
    """
    mask2rle = lambda x: bmask_to_rle(x) if isinstance(x, np.ndarray) and x.dtype == bool and x.ndim == 2 else x
    out = apply_func_to_leaves(prediction, function=mask2rle)

    print_rest = lambda x: print('Remaining arrays:', x.shape) if isinstance(x, np.ndarray) or torch.is_tensor(x) else None
    _ = apply_func_to_leaves(out, function=print_rest)

    with open(path, "w") as fh:
        json.dump(out, fh)

    return path


def load_anno(anno_path):
    """ Read annotation png file.
    """
    anno = Image.open(anno_path)
    assert anno.mode == 'P', "Expecting indexed PNG image"
    anno = np.array(anno)

    instance_ids = np.unique(anno)
    anno_dict = {str(i): bmask_to_rle(anno == i) for i in instance_ids if i != 0 and i != 255}

    return anno_dict


def write_video(output_filepath, frames, fps=30.0):
    """ Write a list of frames to a video file using imageio.
    """
    return imageio.mimwrite(output_filepath, frames, fps=fps)

def cv2_write_video(output_filepath, frames, fps=30.0, fourcc_code='mp4v'):
    """ Write a list of frames to a video file using OpenCV.
    """
    height, width = frames[0].shape[:2]
    is_color = len(frames[0].shape) == 3
    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height), is_color)
    
    for frame in frames:
        out.write(frame[..., ::-1] if is_color else frame)
    out.release()
    return True
        

def load_yaml_file(file_path):
    """ Load a YAML configuration file using OmegaConf.
    """
    return OmegaConf.load(file_path)


def coverage(den_mask, mask2intersect):
    """ Compute the coverage of den_mask by mask2intersect.
    """
    denom = MaskUtils.area(den_mask)
    return MaskUtils.area(MaskUtils.merge([den_mask, mask2intersect], intersect=1)) / denom if denom > 0 else 1


def strip_instance_name(instance_name):
    """ Strip the trailing _1, _2, etc. from an instance name.
    """
    parts = instance_name.split('_')
    if len(parts) > 1 and parts[-1].isdigit():
        return '_'.join(parts[:-1])
    return instance_name


def np_hvstack(x):
    return np.vstack([np.hstack(row) for row in x])

def read_csv(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(line.strip().split(','))
    return data