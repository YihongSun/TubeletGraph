import os.path as osp
import numpy as np
import os, sys, copy, argparse, json
from tqdm import tqdm

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Arrow
import matplotlib.pyplot as plt

sys.path.insert(0, osp.dirname(osp.dirname(__file__)))  # add proj dir to path
from utils import generate_rand_colors, load_yaml_file

class DiagramShape:
    def __init__(self, x, y, shape_type, text, border_color, fill_color=None, width=None, height=None, xscale=180, yscale=90):
        # x and y now represent the center of the shape
        self.x = x * xscale
        self.y = y * yscale
        self.shape_type = shape_type  # 'roundrect' or 'diamond'
        self.text = text
        self.border_color = border_color
        standard_size = {
            'roundrect': (150, 60),
            'diamond': (70, 70)
        }
        standard_width = {
            'roundrect': 150,
            'diamond': 100
        }

        self.width = standard_size[shape_type][0] if width is None else width * xscale
        self.height = standard_size[shape_type][1] if height is None else height * yscale
        self.fill_color = fill_color if fill_color is not None else self.mix_c(border_color)
    
    # Helper methods to get corner coordinates
    def get_left(self):
        return self.x - self.width/2
    
    def get_bottom(self):
        return self.y - self.height/2
    
    def get_right(self):
        return self.x + self.width/2
    
    def get_top(self):
        return self.y + self.height/2
    
    def mix_c(self, color1, color2=(255,255,255), alpha=0.9):
        return tuple(int((1 - alpha) * c1 + alpha * c2) for c1, c2 in zip(color1, color2))

class DiagramArrow:
    def __init__(self, x1, y1, x2, y2, color=(0,0,0)):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color

def calculate_arrow_endpoints(from_shape, to_shape):
    """
    Calculate the endpoints of an arrow from one shape to another,
    connecting at the edges rather than centers for horizontal and vertical arrows.
    """
    from_x, from_y = from_shape.x, from_shape.y
    to_x, to_y = to_shape.x, to_shape.y
    
    # Determine if the arrow is primarily horizontal or vertical
    dx = to_x - from_x
    dy = to_y - from_y
    
    # Set a threshold for what counts as horizontal or vertical
    # If the horizontal distance is much larger than the vertical distance, it's horizontal
    # If the vertical distance is much larger than the horizontal distance, it's vertical
    # Otherwise, it's diagonal and we'll use center-to-center
    
    if dy==0:  # Horizontal arrow
        # From shape: use right or left edge depending on direction
        if dx > 0:  # Arrow points right
            from_x = from_shape.get_right()
        else:  # Arrow points left
            from_x = from_shape.get_left()
        
        # To shape: use left or right edge depending on direction
        if dx > 0:  # Arrow points right
            to_x = to_shape.get_left()
        else:  # Arrow points left
            to_x = to_shape.get_right()
            
    elif dx==0:  # Vertical arrow
        # From shape: use top or bottom edge depending on direction
        if dy > 0:  # Arrow points up
            from_y = from_shape.get_top()
        else:  # Arrow points down
            from_y = from_shape.get_bottom()
        
        # To shape: use bottom or top edge depending on direction
        if dy > 0:  # Arrow points up
            to_y = to_shape.get_bottom()
        else:  # Arrow points down
            to_y = to_shape.get_top()
    
    else:
        raise ValueError("Arrow direction is not horizontal or vertical.")
    
    return from_x, from_y, to_x, to_y

def create_diagram_pdf(shapes, shape_arrows, output_filename, padding=50):
    # Convert shape index arrows to coordinate arrows
    arrows = []
    for from_idx, to_idx, color in shape_arrows:
        from_shape = shapes[from_idx]
        to_shape = shapes[to_idx]
        
        # Calculate proper endpoints for the arrow
        from_x, from_y, to_x, to_y = calculate_arrow_endpoints(from_shape, to_shape)
        
        arrows.append(DiagramArrow(
            from_x, from_y,
            to_x, to_y,
            color=color  # Default color
        ))
    
    # Calculate the bounding box
    min_x = min([shape.get_left() for shape in shapes], default=0)
    min_y = min([shape.get_bottom() for shape in shapes], default=0)
    max_x = max([shape.get_right() for shape in shapes], default=letter[0])
    max_y = max([shape.get_top() for shape in shapes], default=letter[1])
    
    # Also consider arrows in the bounding box
    for arrow in arrows:
        min_x = min(min_x, arrow.x1, arrow.x2)
        min_y = min(min_y, arrow.y1, arrow.y2)
        max_x = max(max_x, arrow.x1, arrow.x2)
        max_y = max(max_y, arrow.y1, arrow.y2)
    
    # Add padding around the bounding box
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding
    
    # Define page dimensions
    page_width = max_x - min_x
    page_height = max_y - min_y
    
    # Create a new PDF with ReportLab using the calculated dimensions
    c = canvas.Canvas(output_filename, pagesize=(page_width, page_height))
    
    # Apply transform to shift everything so (min_x, min_y) becomes the origin
    c.translate(-min_x, -min_y)
    
    # Set up coordinate system (0,0 at bottom-left)
    for shape in shapes[::-1]:
        # Convert RGB tuple to color object
        border_color = colors.Color(shape.border_color[0]/255.0, 
                                  shape.border_color[1]/255.0, 
                                  shape.border_color[2]/255.0)
        fill_color = colors.Color(shape.fill_color[0]/255.0, 
                                shape.fill_color[1]/255.0, 
                                shape.fill_color[2]/255.0)
        
        # Set color
        c.setStrokeColor(border_color)
        c.setFillColor(fill_color)
        c.setLineWidth(2)
        
        # Calculate the top-left corner for drawing (ReportLab uses this coordinate system)
        left_x = shape.get_left()
        bottom_y = shape.get_bottom()
        
        if shape.shape_type == 'roundrect':
            # Draw rounded rectangle
            c.roundRect(left_x, bottom_y, shape.width, shape.height, 10, fill=1)
            
            # Add text
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 12)
            # Text is already centered since x,y is the center
            text_width = c.stringWidth(shape.text)
            c.drawString(shape.x - text_width/2, shape.y - 5, shape.text)
            
        elif shape.shape_type == 'diamond':
            # Draw diamond
            points = [
                (shape.x, shape.get_bottom()),  # bottom
                (shape.get_right(), shape.y),  # right
                (shape.x, shape.get_top()),  # top
                (shape.get_left(), shape.y)  # left
            ]
            p = c.beginPath()
            p.moveTo(points[0][0], points[0][1])
            for point in points[1:]:
                p.lineTo(point[0], point[1])
            p.close()
            c.drawPath(p, fill=1, stroke=1)
            
            # Add text
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 12)
            text_width = c.stringWidth(shape.text)
            c.drawString(shape.x - text_width/2, shape.y - 6, shape.text)
    
    # Draw arrows
    for arrow in arrows:
        arrow_color = colors.Color(arrow.color[0]/255.0, 
                                 arrow.color[1]/255.0, 
                                 arrow.color[2]/255.0)
        c.setStrokeColor(arrow_color)
        c.setLineWidth(2)
        
        # Draw arrow
        c.line(arrow.x1, arrow.y1, arrow.x2, arrow.y2)
        
        # Draw arrowhead
        import math
        angle = math.atan2(arrow.y2 - arrow.y1, arrow.x2 - arrow.x1)
        arrowhead_length = 10
        arrowhead_angle = math.pi / 6
        
        left_x = arrow.x2 - arrowhead_length * math.cos(angle - arrowhead_angle)
        left_y = arrow.y2 - arrowhead_length * math.sin(angle - arrowhead_angle)
        right_x = arrow.x2 - arrowhead_length * math.cos(angle + arrowhead_angle)
        right_y = arrow.y2 - arrowhead_length * math.sin(angle + arrowhead_angle)
        
        c.line(arrow.x2, arrow.y2, left_x, left_y)
        c.line(arrow.x2, arrow.y2, right_x, right_y)
    
    c.save()

def get_obj_color(obj_id, num_colors=10):
    """
    Get the color for a given object ID.
    """
    # Generate a random color based on the object ID
    color = generate_rand_colors(num_colors, lightness=1, seed=10)[obj_id % num_colors]
    return color.tolist()

def get_obj_header(obj):
    obj_header = copy.deepcopy(obj)
    obj_header.y = obj.y + int(0.5 * obj.height)
    obj_header.height = int(0.3 * obj.height)
    obj_header.width = int(0.5 * obj.width)
    obj_header.text = ''
    return obj_header

def create_shape_arrows_from_data(load_data, fps):
    shapes = []
    shape_arrows = []
    for ii, obj_info in enumerate(load_data):
        height = 2 if ii == 0 else 0.85
        obj_color = get_obj_color(int(obj_info['id']))
        obj = DiagramShape(ii, height, 'roundrect', obj_info['desc'], obj_color)
        obj_header = get_obj_header(obj)
        obj_header.text = 't={}s'.format(int(np.round(obj_info['object_start_frame_idx']/fps, 0)))

        shapes.append(obj_header)
        shapes.append(obj)

        if 'action' in obj_info:
            action = DiagramShape(ii, 2, 'diamond', obj_info['action'], (0,0,0), fill_color=(255,255,255))
            shapes.append(action)
            shape_arrows.append((ii*3-2, ii*3+1, get_obj_color(int(load_data[0]['id']))))
            shape_arrows.append((ii*3+1, ii*3-1, obj_color))
    
    final_obj = DiagramShape(ii+1, 2, 'roundrect', obj_info['prior_desc'], get_obj_color(int(load_data[0]['id'])))
    shapes.append(final_obj)
    shape_arrows.append((len(shapes)-2, len(shapes)-1, get_obj_color(int(load_data[0]['id']))))
    return shapes, shape_arrows

def sort_dict_by_value(time_dict):
    sorted_names = sorted(list(time_dict.keys()), key=lambda name: time_dict[name]['object_start_frame_idx'])
    return sorted_names

def get_parser():
    parser = argparse.ArgumentParser(description="Run object tracking methods.")
    parser.add_argument('-c', "--config", default="configs/default.yaml", metavar="FILE", help="path to config file",)
    parser.add_argument('-d', '--dataset', type=str, help='Dataset to run', default='vost')
    parser.add_argument('-s', '--split', type=str, help='Dataset split to run', default='val_instance')
    parser.add_argument('-p', '--pred', type=str, help='prediction directory name', default='vost-val-Annotations_fps5-Ours')
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = load_yaml_file(args.config)
    data_cfg = getattr(cfg.datasets, args.pred.split('-')[0])   # dataset config
    fps = data_cfg.fps

    pred_track_dir = osp.join(cfg.paths.outdir, args.pred)
    instance_names = [x.removesuffix('.json') for x in os.listdir(pred_track_dir) if x.endswith('.json')]

    for instance_name in tqdm(instance_names, desc="Visualizing instances"):
        with open(osp.join(pred_track_dir, instance_name + '.json'), 'r') as f:
            pred_data = json.load(f)
            load_data = pred_data['obj_info']
            
        if len(load_data.keys()) == 0:
            continue
        assert 'object_start_frame_idx' not in load_data['0']
        load_data['0']['object_start_frame_idx'] = 0
        objs_sorted_by_time = sort_dict_by_value(load_data)

        input_data = [load_data[obj_id] | {'id': obj_id} for obj_id in objs_sorted_by_time]
        shapes, shape_arrows = create_shape_arrows_from_data(input_data, fps)

        out_path = osp.join(cfg.paths.visdir, 'state_graph', args.pred, instance_name+'.pdf')
        os.makedirs(osp.dirname(out_path), exist_ok=True)
        create_diagram_pdf(shapes, shape_arrows, out_path, padding=20)