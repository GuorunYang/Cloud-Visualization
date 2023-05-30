import os
import argparse
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pcd_file_py import point_cloud
from tqdm import tqdm
from time import sleep
import visualize_utils_bev as visualize_utils
import visualize_utils_3d as visualize_utils_3d

voxel_size = (0.12, 0.12, 0.2)
# score_thresh = [0, 0.6, 0.45, 0.5, 0.6, 0, 0, 0]       # For Model V0.7
# score_thresh = [0, 0.50, 0.45, 0.50, 0.50, 0, 0, 0]    # For Model V0.8
# score_thresh = [0, 0.64, 0.43, 0.55, 0.64]      # For Model V2.3
# score_thresh = [0, 0.58, 0.44, 0.53, 0.63]      # For Model V2.5
score_thresh = [0, 0.62, 0.43, 0.55, 0.62]      # For Model V2.6

v2x_region = [0, 40, -50, 50]

# Classifications and color map
cls_array = [
    'Car',
    'Pedestrian',
    'Cyclist',
    'Truck',
    'Cone',
    'Unknown',
    'Dontcare',
    'Traffic_Warning_Object',
    'Traffic_Warning_Sign',
    'Road_Falling_Object',
    'Road_Intrusion_Object',
    'Animal'
]
cls_dict = {
    '0': 'Car',
    '1': 'Pedestrian',
    '2': 'Cyclist',
    '3': 'Truck',
    '4': 'Cone',
    '5': 'Unknown',
    '6': 'Dontcare',
    '7': 'Traffic_Warning_Object',
    '8': 'Traffic_Warning_Sign',
    '9': 'Road_Falling_Object',
    '10': 'Road_Intrusion_Object',
    '11': 'Animal'
}
colormap = [
    [0, 0, 0],
    [68, 255, 117],     # 0 Car: Green
    # [255, 151, 45],   # 1 Pedestrian: Dark Orange
    [255, 51, 51],      # 1 Pedestrian: Red
    [255, 204, 45],     # 2 Cyclist: Gold Orange
    [142, 118, 255],    # 3 Truck: Purple
    [238, 238, 0],      # 4 Cone: Yellow
    [224, 224, 224],    # 5 Unknown: Light Grey
    [190, 190, 190],    # 6 DontCare: Grey
    [255, 215, 0],      # 7 Traffic_Warning_Object: Gold
    [255, 192, 203],    # 8 Traffic_Warning_Sign: Pink
    [255, 127, 36],    # 9 Road_Falling_Object: Chocolate1
    [255, 64, 64],    # 10 Road_Intrusion_Object: Brown1
    [255, 0, 255],    # 11 Animal: Magenta
]

def check_args(cloud, result = None, label = None, poly = None, voxel = None, image = None):
    '''
        Check the input arguments and set the status
            1. draw frame or draw sequence
            2. draw cloud, draw result, draw label, draw poly, draw voxel, or draw image
    '''

    # Step 0: Define the Arguments
    draw_frame, draw_sequence = False, False
    draw_cloud, draw_result, draw_label, draw_poly, draw_voxel, draw_image = False, False, False, False, False, False
    bin_cloud, pcd_cloud, txt_cloud = False, False, False
    cloud_list = []
    # Step 1: Check the cloud
    if os.path.isdir(cloud):
        draw_sequence = True
        cloud_list = sorted(os.listdir(cloud))
        for i in range(len(cloud_list)):
            if not (cloud_list[i].endswith('.bin') or cloud_list[i].endswith('.pcd') or cloud_list[i].endswith('.txt')):
                warnings.warn('The Cloud: ', cloud_list[i], ' is neither pcd file nor bin file')
                break
            if i == len(cloud_list) - 1:
                # Reach the end of cloud list
                if cloud_list[i].endswith('.bin'):
                    bin_cloud = True
                elif cloud_list[i].endswith('pcd'):
                    pcd_cloud = True
                elif cloud_list[i].endswith('txt'):
                    txt_cloud = True
                draw_cloud = True
    elif os.path.isfile(cloud):
        draw_frame = True
        if cloud.endswith('.bin'):
            bin_cloud = True
            draw_cloud = True
            cloud_list.append(cloud)
        elif cloud.endswith('pcd'):
            pcd_cloud = True
            draw_cloud = True
            cloud_list.append(cloud)
        elif cloud.endswith('.txt'):
            txt_cloud = True
            draw_cloud = True
        else:
            warnings.warn('The Cloud is neither pcd file nor bin file')

    # Step 2: Check the result
    if result is not None:
        if draw_sequence and os.path.isdir(result):
            result_list = sorted(os.listdir(result))
            if len(result_list) == len(cloud_list):
                draw_result = True
            else:
                warnings.warn('The results are not matching the clouds')
        elif draw_frame and os.path.isfile(result):
            draw_result = True
        else:
            warnings.warn('The results are not matching the clouds')

    # Step 3: Check the label
    if label is not None:
        if draw_sequence and os.path.isdir(label):
            label_list = sorted(os.listdir(label))
            if len(label_list) == len(cloud_list):
                draw_label = True
            else:
                warnings.warn('The labels are not matching the clouds')
        elif draw_frame and os.path.isfile(label):
            draw_label = True
        else:
            warnings.warn('The labels are not matching the clouds')

    # Step 4: Check the poly
    if poly is not None:
        if draw_sequence and os.path.isdir(poly):
            poly_list = sorted(os.listdir(poly))
            if len(poly_list) == len(cloud_list):
                draw_poly = True
            else:
                warnings.warn('The polygon results are not matching the clouds')
        elif draw_frame and os.path.isfile(poly):
            draw_poly = True
        else:
            warnings.warn('The polygon results are not matching the clouds')

    # Step 5: Check the voxel
    if voxel is not None:
        if draw_sequence and os.path.isdir(voxel):
            voxel_list = sorted(os.listdir(voxel))
            if len(voxel_list) == len(cloud_list):
                draw_voxel = True
            else:
                warnings.warn('The voxel maps are not matching the clouds')
        elif draw_frame and os.path.isfile(voxel):
            draw_voxel = True
        else:
            warnings.warn('The voxel maps are not matching the clouds')

    # Step 6: Check the image
    if image is not None:
        if draw_sequence and os.path.isdir(image):
            image_list = sorted(os.listdir(image))
            if len(image_list) == len(cloud_list):
                draw_image = True
            else:
                warnings.warn('The images are not matching the clouds')
        elif draw_frame and os.path.isfile(image):
            draw_image = True
        else:
            warnings.warn('The images are not matching the clouds')


    return draw_frame, draw_sequence, draw_cloud, draw_result, draw_label, draw_poly, draw_voxel, draw_image


def read_cloud(cloud_path, with_ring = False):
    if cloud_path.endswith('.bin'):
        if not with_ring:
            cloud = np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 5)
            return cloud
        else:
            cloud = np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 5)
            return cloud
    elif cloud_path.endswith('.pcd'):
        # if(not pcd_with_ring):
        #     cloud = point_cloud.load_pcd_from_path(cloud_path, from_bag=True)
        # else:
        #     cloud=load_pcd_with_ring(cloud_path)
        cloud = point_cloud.load_pcd_from_path(cloud_path, from_bag = True, include_ring = with_ring)
        return cloud
    elif cloud_path.endswith('.txt'):
        cloud = np.loadtxt(cloud_path, dtype = np.float32)
        return cloud
    else:
        raise TypeError('Cloud extension is not .bin or .pcd')


def load_single_result(det_path):
    '''
        Load detection results from single frame
    '''
    results = [{'dets': {}} for i in range(1)]
    with open(det_path, 'r') as f:
        det_lines = f.readlines()
        for det_ln in det_lines:
            det_type, det_box, det_score, det_iou, det_track = load_line(det_ln)
            if 'det_box' in results[0]['dets']:
                results[0]['dets']['det_box'].append(det_box)
            else:
                results[0]['dets']['det_box'] = [det_box]
            if 'name' in results[0]['dets']:
                results[0]['dets']['name'].append(det_type)
            else:
                results[0]['dets']['name'] = [det_type]
            if 'score' in results[0]['dets']:
                results[0]['dets']['score'].append(det_score)
            else:
                results[0]['dets']['score'] = [det_score]
            if det_track is not None:
                if 'track_info' in labels[0]['dets']:
                    labels[0]['dets']['track_info'].append(det_track)
                else:
                    labels[0]['dets']['track_info'] = [det_track]

    for i in range(len(results)):
        for key in results[i]['dets']:
            results[i]['dets'][key] = np.array(results[i]['dets'][key])
    return results


def load_single_poly(poly_path):
    '''
        Load polygon results from single frame
    '''
    poly_results = [{'polys': {}} for i in range(1)]
    with open(poly_path, 'r') as f:
        poly_lines = f.readlines()
        for poly_ln in poly_lines:
            poly_type, poly_box, track_info, poly_vertices = load_poly_line(poly_ln)
            if 'poly_box' in poly_results[0]['polys']:
                poly_results[0]['polys']['poly_box'].append(poly_box)
            else:
                poly_results[0]['polys']['poly_box'] = [poly_box]
            if 'name' in poly_results[0]['polys']:
                poly_results[0]['polys']['name'].append(poly_type)
            else:
                poly_results[0]['polys']['name'] = [poly_type]
            if 'poly_vertices' in poly_results[0]['polys']:
                poly_results[0]['polys']['poly_vertices'].append(poly_vertices)
            else:
                poly_results[0]['polys']['poly_vertices'] = [poly_vertices]
            if track_info is not None:
                if 'track_info' in poly_results[0]['polys']:
                    poly_results[0]['polys']['track_info'].append(track_info)
                else:
                    poly_results[0]['polys']['track_info'] = [track_info]
    for i in range(len(poly_results)):
        for key in poly_results[i]['polys']:
            poly_results[i]['polys'][key]=np.asarray(poly_results[i]['polys'][key])
    return poly_results


def load_single_label(label_path):
    '''
        Load labels from single frame
    '''
    labels = [{'annos': {}} for i in range(1)]
    with open(label_path, 'r') as f:
        label_lines = f.readlines()
        for label_ln in label_lines:
            label_type, label_box, label_score, label_iou, label_track = load_line(label_ln)
            if 'gt_box' in labels[0]['annos']:
                labels[0]['annos']['gt_box'].append(label_box)
            else:
                labels[0]['annos']['gt_box'] = [label_box]
            if 'name' in labels[0]['annos']:
                labels[0]['annos']['name'].append(label_type)
            else:
                labels[0]['annos']['name'] = [label_type]
            if label_track is not None:
                if 'track_info' in labels[0]['annos']:
                    labels[0]['annos']['track_info'].append(label_track)
                else:
                    labels[0]['annos']['track_info'] = [label_track]

    for i in range(len(labels)):
        for key in labels[i]['annos']:
            labels[i]['annos'][key] = np.array(labels[i]['annos'][key])
    return labels

def get_cloud_list(cloud_dir, sort_by_num = False):
    cloud_list = os.listdir(cloud_dir)
    if sort_by_num:
        cloud_list = sorted(cloud_list, key=lambda x: int(x.split('.')[0]))
    else:
        cloud_list = sorted(cloud_list)
    return cloud_list


def get_voxel_list(voxel_dir, sort_by_num = False):
    voxel_list = os.listdir(voxel_dir)
    if sort_by_num:
        voxel_list = sorted(voxel_list, key=lambda x: int(x.split('.')[0]))
    else:
        voxel_list = sorted(voxel_list)
    return voxel_list


def get_image_list(image_dir, sort_by_num = False):
    image_list = os.listdir(image_dir)
    if sort_by_num:
        image_list = sorted(image_list, key=lambda x: int(x.split('.')[0]))
    else:
        image_list = sorted(image_list)
    return image_list


def load_poly_line(poly_ln):
    '''
        The format of poly results:
            object_id   type    length  width   height  x   y   z   direction   vx  vy  vz
            timestamp   is_static   frame_id    track_conf  first_track  is_detection  points_number
            polygon_height ...(x,y,z)... Polygon
    '''
    line_array = poly_ln.strip().split()
    poly_type = None        # Poly Type
    poly_box = None         # Poly Box: locations, dimensions, rotation_y
    poly_track = None       # Poly Track: object id, vx, vy, vz
    poly_vertices = None    # Poly Vertices
    if len(line_array) >= 18:
        object_id = [float(line_array[0])]
        if line_array[1] in cls_array:
            poly_type = line_array[1]
        else:
            poly_type = cls_dict[line_array[1]]
        dimensions = [float(line_array[3]), float(line_array[2]), float(line_array[4])]     # w, l, h
        locations = [float(line_array[5]), float(line_array[6]), float(line_array[7])]
        rotation_y = [float(line_array[8])]
        velocity = [float(line_array[9]), float(line_array[10]), float(line_array[11])]
        poly_box = locations + dimensions + rotation_y
        poly_track = object_id + velocity
        poly_height = float(poly_ln.split()[19])
        poly_vertices_bottom = [float(i) for i in poly_ln.split()[20:-1]]
        poly_vertices_up = poly_vertices_bottom.copy()
        for i in range(0, len(poly_vertices_up), 3):
            poly_vertices_up[i + 2] += poly_height
        poly_vertices = poly_vertices_bottom + poly_vertices_up
    return poly_type, poly_box, poly_track, poly_vertices


def load_line(ln):
    '''
        There are several formats for result line.
        We can identify the formats by the length of result line.
        1. Old Result Version (Length = 9):
            type    width   length  height  x       y       z       rotation_y  score

        2. Old Label Version (Length = 11):
            type    occlusion   truncation  alpha  length   width  height  x       y       z       rotation_y

        3. KITTI Standard Label Version (Length = 15):
            type    occlusion   truncation  alpha   left    top     right       bottom
            height  width       length      x       y       z       rotation_y

        4. KITTI Standard Result Version (Length = 16):
            type    occlusion   truncation  alpha   left    top     right       bottom
            height  width       length      x       y       z       rotation_y  score

        5. KITTI FP Result Version (Length = 20, for false positive):
            type    occlusion   truncation  alpha   left    top     right   bottom  height  width   length
            x       y       z   rotation_y  score   2d_iou  bev_iou 3d_iou  center_distance

        6. KITTI FN Label Version (Length = 19, for false negative):
            type    occlusion   truncation  alpha   left    top     right   bottom height  width    length
            x       y       z   rotation_y  2d_iou  bev_iou 3d_iou  center_distance

        7. MOT Label (Length = 14)
            object_id   type    length  width       height  x   y   z   direction   vx  vy  vz
            timestamp   is_static

        8. MOT Result (Length = 18)
            object_id   type    length  width       height  x   y   z   direction   vx  vy  vz
            timestamp   is_static       frame_id    track_conf  first_track is_detection
    '''
    line_array = ln.strip().split()
    type, box3d, score, ious = None, None, None, None
    track_info = None           # Track info includes: object id, vx, vy, vz
    if len(line_array) == 9:
        # 1. Old Result Version (Length = 9):
        type        = line_array[0]
        dimensions  = [float(line_array[1]), float(line_array[2]), float(line_array[3])]
        locations   = [float(line_array[4]), float(line_array[5]), float(line_array[6])]
        rotation_y  = [float(line_array[7])]
        score       = float(line_array[8])
        box3d       = locations + dimensions + rotation_y
    elif len(line_array) == 11:
        # 2. Old label Version (Length = 11):
        type = line_array[0]
        dimensions = [float(line_array[5]), float(line_array[4]), float(line_array[6])]
        locations = [float(line_array[7]), float(line_array[8]), float(line_array[9])]
        rotation_y = [float(line_array[10])]
        box3d = locations + dimensions + rotation_y
    elif len(line_array) == 15:
        # 3. KITTI Standard Label Version (Length = 15):
        type        = line_array[0]
        dimensions  = [float(line_array[9]), float(line_array[10]), float(line_array[8])]
        locations   = [float(line_array[11]), float(line_array[12]), float(line_array[13])]
        rotation_y  = [float(line_array[14])]
        box3d       = locations + dimensions + rotation_y
    elif len(line_array) == 16:
        # 4. KITTI Standard Result Version (Length = 16):
        type        = line_array[0]
        dimensions  = [float(line_array[9]), float(line_array[10]), float(line_array[8])]
        locations   = [float(line_array[11]), float(line_array[12]), float(line_array[13])]
        rotation_y  = [float(line_array[14])]
        score       = float(line_array[15])
        box3d       = locations + dimensions + rotation_y
    elif len(line_array) == 20:
        # 5. KITTI FP Result Version (Length = 20, for false positive):
        type        = line_array[0]
        dimensions  = [float(line_array[9]), float(line_array[10]), float(line_array[8])]
        locations   = [float(line_array[11]), float(line_array[12]), float(line_array[13])]
        rotation_y  = [float(line_array[14])]
        score       = float(line_array[8])
        ious        = [float(line_array[16]), float(line_array[17]), float(line_array[18]), float(line_array[19])]
        box3d       = locations + dimensions + rotation_y
    elif len(line_array) == 19:
        # 6. KITTI FN Label Version (Length = 19, for false negative):
        type = line_array[0]
        dimensions = [float(line_array[9]), float(line_array[10]), float(line_array[8])]
        locations = [float(line_array[11]), float(line_array[12]), float(line_array[13])]
        rotation_y = [float(line_array[14])]
        ious = [float(line_array[15]), float(line_array[16]), float(line_array[17]), float(line_array[18])]
        box3d = locations + dimensions + rotation_y
    elif len(line_array) == 14 or len(line_array) == 17 or len(line_array) == 18:
        # 7. MOT Label Version (Length = 18)
        if line_array[1] in cls_array:
            type = line_array[1]
        else:
            type = cls_dict[line_array[1]]
        dimensions = [float(line_array[3]), float(line_array[2]), float(line_array[4])]
        locations = [float(line_array[5]), float(line_array[6]), float(line_array[7])]
        rotation_y = [float(line_array[8])]
        object_id = [float(line_array[0])]
        if len(line_array) == 18:
            score = float(line_array[16])
        else:
            score = 1.0
        velocity = [float(line_array[9]), float(line_array[10]), float(line_array[11])]
        box3d = locations + dimensions + rotation_y
        track_info = object_id + velocity
    return type, box3d, score, ious, track_info


def load_polys(poly_dir, sort_by_num = False):
    poly_list = os.listdir(poly_dir)
    if sort_by_num:
        poly_list = sorted(poly_list, key=lambda x: int(x.split('.')[0]))
    else:
        poly_list = sorted(poly_list)
    max_frames = len(poly_list)
    poly_results = [{'polys': {}} for i in range(max_frames)]
    for frame_id, poly_fn in enumerate(poly_list):
        poly_path = os.path.join(poly_dir, poly_fn)
        with open(poly_path, 'r') as f:
            poly_lines = f.readlines()
            for poly_ln in poly_lines:
                poly_type, poly_box, track_info, poly_vertices = load_poly_line(poly_ln)
                if 'poly_box' in poly_results[frame_id]['polys']:
                    poly_results[frame_id]['polys']['poly_box'].append(poly_box)
                else:
                    poly_results[frame_id]['polys']['poly_box'] = [poly_box]
                if 'name' in poly_results[frame_id]['polys']:
                    poly_results[frame_id]['polys']['name'].append(poly_type)
                else:
                    poly_results[frame_id]['polys']['name'] = [poly_type]
                if 'poly_vertices' in poly_results[frame_id]['polys']:
                    poly_results[frame_id]['polys']['poly_vertices'].append(poly_vertices)
                else:
                    poly_results[frame_id]['polys']['poly_vertices'] = [poly_vertices]
                if track_info is not None:
                    if 'track_info' in poly_results[frame_id]['polys']:
                        poly_results[frame_id]['polys']['track_info'].append(track_info)
                    else:
                        poly_results[frame_id]['polys']['track_info'] = [track_info]

    for i in range(len(poly_results)):
        for key in poly_results[i]['polys']:
            poly_results[i]['polys'][key] = np.array(poly_results[i]['polys'][key])
    return poly_results


def load_results(det_dir, sort_by_num = False):
    det_list = os.listdir(det_dir)
    if sort_by_num:
        det_list = sorted(det_list, key=lambda x: int(x.split('.')[0]))
    else:
        det_list = sorted(det_list)
    max_frames = len(det_list)
    results = [{'dets': {}} for i in range(max_frames)]
    for frame_id, det_fn in enumerate(det_list):
        det_path = os.path.join(det_dir, det_fn)
        with open(det_path, 'r') as f:
            det_lines = f.readlines()
            for det_ln in det_lines:
                det_type, det_box, det_score, det_iou, det_track = load_line(det_ln)
                if 'det_box' in results[frame_id]['dets']:
                    results[frame_id]['dets']['det_box'].append(det_box)
                else:
                    results[frame_id]['dets']['det_box'] = [det_box]
                if 'name' in results[frame_id]['dets']:
                    results[frame_id]['dets']['name'].append(det_type)
                else:
                    results[frame_id]['dets']['name'] = [det_type]
                if 'score' in results[frame_id]['dets']:
                    results[frame_id]['dets']['score'].append(det_score)
                else:
                    results[frame_id]['dets']['score'] = [det_score]
                if det_track is not None:
                    if 'track_info' in results[frame_id]['dets']:
                        results[frame_id]['dets']['track_info'].append(det_track)
                    else:
                        results[frame_id]['dets']['track_info'] = [det_track]

    for i in range(len(results)):
        for key in results[i]['dets']:
            results[i]['dets'][key] = np.array(results[i]['dets'][key])
    return results


def load_labels(label_dir, sort_by_num = False):
    label_list = os.listdir(label_dir)
    if sort_by_num:
        label_list = sorted(label_list, key=lambda x: int(x.split('.')[0]))
    else:
        label_list = sorted(label_list)
    max_frames = len(label_list)
    labels = [{'annos': {}} for i in range(max_frames)]
    for frame_id, label_fn in enumerate(label_list):
        label_path = os.path.join(label_dir, label_fn)
        with open(label_path, 'r') as f:
            label_lines = f.readlines()
            for label_ln in label_lines:
                label_type, label_box, label_score, label_iou, label_track = load_line(label_ln)
                if 'gt_box' in labels[frame_id]['annos']:
                    labels[frame_id]['annos']['gt_box'].append(label_box)
                else:
                    labels[frame_id]['annos']['gt_box'] = [label_box]
                if 'name' in labels[frame_id]['annos']:
                    labels[frame_id]['annos']['name'].append(label_type)
                else:
                    labels[frame_id]['annos']['name'] = [label_type]
                if label_track is not None:
                    if 'track_info' in labels[frame_id]['annos']:
                        labels[frame_id]['annos']['track_info'].append(label_track)
                    else:
                        labels[frame_id]['annos']['track_info'] = [label_track]
    for i in range(len(labels)):
        for key in labels[i]['annos']:
            labels[i]['annos'][key] = np.array(labels[i]['annos'][key])
    return labels


def draw_3d_polys(fig, vertices3d, thickness=3, color=(224, 224, 224)):
    if vertices3d is None or len(vertices3d) == 0:
        return fig
    cur_color = tuple(np.asarray(color, dtype = np.float) / 255.0)
    fig = visualize_utils_3d.draw_poly3d(vertices3d, fig, poly_color=cur_color, poly_width=thickness)
    return fig

def draw_3d_boxes(fig, boxes3d, labels, scores=None, track_ids=None, thickness=3,
                  colorize_with_label = True, color=(0, 255, 0)):
    if boxes3d is None or boxes3d.shape[0] == 0:
        return fig
    corners3d = visualize_utils.boxes3d_to_corners3d_lidar(boxes3d)
    for cur_label in range(labels.min(), labels.max() + 1):
        cur_color = tuple(np.array(colormap[cur_label]) / 255)
        # Filter the boxes by score
        if scores is None:
            mask = (labels == cur_label)
        else:
            mask = (labels == cur_label) & (scores > score_thresh[cur_label])
        if mask.sum() == 0:
            continue
        # Draw 3D boxes
        if colorize_with_label:
            fig = visualize_utils_3d.draw_box3d(corners3d[mask], fig, box_color=cur_color, box_width=thickness)
        else:
            fig = visualize_utils_3d.draw_box3d(corners3d[mask], fig, box_color=tuple(color), box_width=thickness)
        # Draw tracking information
        if track_ids is not None:
            masked_boxes3d = boxes3d[mask]
            masked_corners3d = corners3d[mask]
            masked_track_ids = track_ids[mask]
            fig = visualize_utils_3d.draw_tracksids(masked_boxes3d, masked_corners3d, masked_track_ids, fig,
                                                    arrow_color=cur_color, arrow_width=thickness)
    return fig


def draw_bev_polys(bev_map, polys, thickness = 2, color = (224, 224, 224),
                   area_scope =[[-72, 92], [-72, 72], [-5, 5]]):
    if polys is None or len(polys) == 0:
        return bev_map
    else:
        bev_map = visualize_utils.draw_bev_polys(image = bev_map, polys = polys, color=color, thickness=thickness,
                                                 voxel_size=voxel_size, area_scope=area_scope)
        return bev_map


def draw_bev_boxes(bev_map, boxes3d, labels, scores=None, track_ids=None, thickness=2,
                   colorize_with_label = True, color=(0, 255, 0),
                   area_scope =[[-72, 92], [-72, 72], [-5, 5]]):
    classes = ['', 'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Cone', 'Unknown', 'Dontcare']
    if boxes3d is None or boxes3d.shape[0] == 0:
        return bev_map
    corners3d = visualize_utils.boxes3d_to_corners3d_lidar(boxes3d)
    bev_corners = visualize_utils.corners3d_to_bev_corners(corners3d, voxel_size=voxel_size, area_scope=area_scope)

    for cur_label in range(labels.min(), labels.max() + 1):
        if scores is None:
            mask = (labels == cur_label)
        else:
            mask = (labels == cur_label) & (scores > score_thresh[cur_label])
        if mask.sum() == 0:
            continue
        masked_track_ids = None
        if track_ids is not None:
            masked_track_ids = track_ids[mask]
        if colorize_with_label:
            bev_color_map = np.asarray(colormap)
            bev_color_map = bev_color_map[:, [2, 1, 0]]
            if scores is not None:
                bev_map = visualize_utils.draw_bev_boxes(bev_map, bev_corners[mask],
                                                        color=tuple(bev_color_map[cur_label]),
                                                        scores=scores[mask], thickness=thickness, track_ids=masked_track_ids,
                                                        box_labels=[classes[cur_label]] * mask.sum())
            else:
                bev_map = visualize_utils.draw_bev_boxes(bev_map, bev_corners[mask],
                                                        color=tuple(bev_color_map[cur_label]),
                                                        scores=None, thickness=thickness, track_ids=masked_track_ids,
                                                        box_labels=[classes[cur_label]] * mask.sum())
        else:
            if scores is not None:
                bev_map = visualize_utils.draw_bev_boxes(bev_map, bev_corners[mask], color=tuple(color),
                                                        scores=scores[mask], thickness=thickness, track_ids=masked_track_ids,
                                                        box_labels=[classes[cur_label]] * mask.sum())
            else:
                bev_map = visualize_utils.draw_bev_boxes(bev_map, bev_corners[mask], color=tuple(color),
                                                        scores=None, thickness=thickness, track_ids=masked_track_ids,
                                                        box_labels=[classes[cur_label]] * mask.sum())
        # bev_maps = visualize_utils.draw_bev_circle_scale(bev_maps, voxel_size=voxel_size, area_scope=area_scope)
        # bev_maps = visualize_utils.draw_bev_det_scope(bev_maps, det_scope=det_scope, voxel_size=voxel_size, area_scope=area_scope)
    return bev_map


def count_number(frame_results = None, frame_labels = None, view_region = [0, 40, -50, 50]):
    count_dict = {
        'Car': 0, 'Truck' : 1, 'Pedestrian' : 2, 'Cyclist' : 3
    }
    count_array = np.zeros((4, ), dtype = np.int)
    if frame_results is not None:
        det_types = None if 'name' not in frame_results['dets'] else frame_results['dets']['name']
        det_boxes = None if 'det_box' not in  frame_results['dets'] else frame_results['dets']['det_box']
        if not det_types is None:
            for i, cls in enumerate(det_types):
                cls_id = count_dict[cls]
                x, y = det_boxes[i][0], det_boxes[i][2]
                if x < view_region[0] or x > view_region[1] or y < view_region[2] or y > view_region[3]:
                    continue
                count_array[cls_id] += 1
    elif frame_labels is not None:
        gt_types = None if 'name' not in frame_labels['annos'] else frame_labels['annos']['name']
        gt_boxes = None if 'gt_box' not in frame_labels['annos'] else frame_labels['annos']['gt_box']
        if not gt_types is None:
            for i, cls in enumerate(gt_types):
                cls_id = count_dict[cls]
                x, y = gt_boxes[i][0], gt_boxes[i][2]
                if x < view_region[0] or x > view_region[1] or y < view_region[2] or y > view_region[3]:
                    continue
                count_array[cls_id] += 1
    return count_array


def get_polys(frame_polys = None):
    if frame_polys is not None:
        classes = ['', 'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Cone', 'Unknown', 'Dontcare']
        poly_boxes3d = None if 'poly_box' not in frame_polys['polys'] else frame_polys['polys']['poly_box']
        poly_vertices = None if 'poly_vertices' not in frame_polys['polys'] else frame_polys['polys']['poly_vertices']
        poly_types = None if 'name' not in frame_polys['polys'] else frame_polys['polys']['name']
        poly_trackids = None if 'track_info' not in frame_polys['polys'] else frame_polys['polys']['track_info']
        poly_labels = None if poly_types is None else np.array([classes.index(type) for type in poly_types if type != None])
        return poly_boxes3d, poly_vertices, poly_labels, poly_trackids
    else:
        return None


def get_boxes_from_results(frame_results = None):
    if frame_results is not None:
        classes = ['', 'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Cone', 'Unknown', 'Dontcare',
                   'Traffic_Warning_Object', 'Traffic_Warning_Sign',
                   'Road_Falling_Object', 'Road_Intrusion_Object', 'Animal'
                   ]
        det_boxes3d = None if 'det_box' not in frame_results['dets'] else frame_results['dets']['det_box']
        det_scores = None if 'score' not in frame_results['dets'] else frame_results['dets']['score']
        det_types = None if 'name' not in frame_results['dets'] else frame_results['dets']['name']
        det_trackids = None if 'track_info' not in frame_results['dets'] else frame_results['dets']['track_info']
        det_labels = None if det_types is None else np.array([classes.index(type) for type in det_types if type != None])
        return det_boxes3d, det_scores, det_labels, det_trackids
    else:
        return None

def get_boxes_from_labels(frame_labels = None):
    if frame_labels is not None:
        classes = ['', 'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Cone', 'Unknown', 'Dontcare',
                   'Traffic_Warning_Object', 'Traffic_Warning_Sign',
                   'Road_Falling_Object', 'Road_Intrusion_Object', 'Animal'
                  ]
        gt_boxes3d = None if 'gt_box' not in frame_labels['annos'] else frame_labels['annos']['gt_box']
        gt_types = None if 'name' not in frame_labels['annos'] else  frame_labels['annos']['name']
        gt_scores = None if 'score' not in frame_labels['annos'] else frame_labels['annos']['score']
        gt_trackids = None if 'track_info' not in frame_labels['annos'] else frame_labels['annos']['track_info']
        gt_labels = None if gt_types is None else np.array([classes.index(type) for type in gt_types])
        return gt_boxes3d, gt_scores, gt_labels, gt_trackids
    else:
        return None


def draw_3d_map(cloud_path, frame_results = None, frame_labels = None, frame_polys = None, voxel_path = None, image_path = None,
                draw_cloud = True, draw_result = False, draw_label = False, draw_poly = False, draw_voxel = False, draw_image = False,
                draw_ground = True, cloud_with_ring = False, v2x = False, use_screenshot = False, debug_mode = False):
    import mayavi.mlab as mlab
    if draw_cloud:
        cloud = read_cloud(cloud_path, cloud_with_ring)
        fig = visualize_utils_3d.draw_lidar(cloud)
        if draw_result and draw_label:
            gt_color    = (178/255.0, 255/255.0, 102/255.0)
            det_color   = (255/255.0, 151/255.0, 45/255.0)
            det_boxes3d, det_scores, det_cls, det_trackids = get_boxes_from_results(frame_results)
            gt_boxes3d, gt_scores, gt_cls, gt_trackids = get_boxes_from_labels(frame_labels)
            fig = draw_3d_boxes(fig, boxes3d=det_boxes3d, labels=det_cls,
                                scores=det_scores, track_ids=det_trackids,
                                colorize_with_label = False, color=det_color, thickness = 2)
            fig = draw_3d_boxes(fig, boxes3d=gt_boxes3d, labels=gt_cls,
                                scores=gt_scores, track_ids=gt_trackids,
                                colorize_with_label = False, color=gt_color, thickness = 2)
        else:
            if draw_label:
                gt_boxes3d, gt_scores, gt_cls, gt_trackids = get_boxes_from_labels(frame_labels)
                fig = draw_3d_boxes(fig, boxes3d=gt_boxes3d, labels=gt_cls, scores=gt_scores, track_ids=gt_trackids)
            if draw_result:
                det_boxes3d, det_scores, det_cls, det_trackids = get_boxes_from_results(frame_results)
                fig = draw_3d_boxes(fig, boxes3d=det_boxes3d, labels=det_cls, scores=det_scores, track_ids=det_trackids)
            if draw_poly:
                poly_boxes3d, poly_vertices3d, poly_cls, poly_trackids = get_polys(frame_polys)
                fig = draw_3d_polys(fig, vertices3d = poly_vertices3d)
            if draw_voxel:
                # voxel_color = tuple(np.array([0, 0, 244]) / 255)
                voxel_color = tuple(np.array([216, 216, 216]) / 255)
                fig = visualize_utils_3d.draw_3d_voxels(fig, height_offset = -1.4, voxel_height = 1.7,
                    voxel_path = voxel_path, voxel_color = voxel_color)
        if draw_ground:
            fig = visualize_utils_3d.draw_ground_plane(fig)

        if not v2x:
            # mlab.view(azimuth=179.84, elevation=65.03, distance=95.96,
            #           focalpoint=np.array([13.19, -0.48, 4.50]), roll=89.99)
            if draw_voxel:
                # mlab.view(azimuth=175.36, elevation=25.08, distance=170.00,
                #           focalpoint=np.array([10.70, 0.15, -0.05]), roll=89.99)
                mlab.view(azimuth=179.11, elevation=61.24, distance=79.31,
                          focalpoint=np.array([10.70, 0.15, -0.05]), roll=89.99)
            else:
                mlab.view(azimuth=179.11, elevation=61.24, distance=79.31,
                          focalpoint=np.array([10.70, 0.15, -0.05]), roll=89.99)
        else:
            # Viewpoint for V2X
            mlab.view(azimuth=177.03, elevation=41.72, distance=116.11,
                      focalpoint=np.array([20.42, 1.71, 16.32]), roll=89.99)

        if debug_mode:
            mlab.show()
            return fig
        elif use_screenshot or draw_image:
            from pyface.api import GUI
            GUI().process_events()
            img = mlab.screenshot()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if use_screenshot:
                if not v2x:
                    me_points = np.array([[0.0, 0.0, 1.6]], dtype=np.float32)
                    me_vertex = visualize_utils_3d.proj_3d_to_2d(me_points, fig)[0]
                    me_vertex = np.asarray(me_vertex, dtype=np.int)
                    img = visualize_utils.draw_me(img, me_vertex, './examples/icons/car_2d.png', 0.19, 65)  # For Car
                    img = visualize_utils.draw_legend(img, [1350, 50], './examples/icons/legends_car.png', 1.0)
                else:
                    count_colormap = [
                        [117, 255, 68, 0],  # Car: Green
                        [255, 118, 142, 0],  # Truck: Purple
                        [45, 151, 255, 0],  # Pedestrian: Dark Orange
                        [45, 204, 255, 0],  # Cyclist: Gold Orange
                    ]
                    count_array = count_number(frame_results, frame_labels, view_region=v2x_region)
                    img = visualize_utils.draw_legend(img, [1350, 63], './examples/icons/legends_v2x.png', 1.0)
                    img = visualize_utils.draw_counts(img, count_number = count_array,
                        count_color = count_colormap, count_vertex = [1405, 54])
            if draw_image:
                img = visualize_utils.draw_image(img, image_path, 0.28)
            mlab.close()
            return img
        else:
            return fig

def draw_bev_map(cloud_path, frame_results = None, frame_labels = None, frame_polys = None,
                 frame_voxel_path = None, frame_image_path = None, draw_cloud = True, draw_result = False,
                 draw_label = False, draw_poly = False, draw_voxel = False, draw_image = False,
                 with_ring = False, area_scope = [[-72, 92], [-72, 72], [-5, 5]],
                 colorize_by_intensity = False, histogram_intensity = True):
    if draw_cloud:
        cloud = read_cloud(cloud_path, with_ring)
        voxel_map = visualize_utils.convert_pts_to_bev_map(cloud, voxel_size=voxel_size, area_scope=area_scope)
        bev_map = voxel_map.sum(axis=2)

        if colorize_by_intensity:
            # Get the intensity map from last channel
            intensity_map = voxel_map[:, :, -1]
            if histogram_intensity:
                intensity_map = intensity_map.astype(np.uint8)
                intensity_map = cv2.equalizeHist(intensity_map)
            else:
                intensity_map = intensity_map / max(intensity_map.max(), 1.0)

            cmap = plt.get_cmap('hot')
            bev_map = cmap(intensity_map)
            bev_map = np.delete(bev_map, 3, axis = 2)
            bev_map[:, :, [0, 1, 2]] = bev_map[:, :, [2, 1, 0]] * 255
            bev_map = bev_map.astype(np.uint8)

        else:
            bev_index = bev_map > 0
            bev_map = np.zeros([bev_map.shape[0], bev_map.shape[1], 3], dtype = np.uint8)
            bev_map[bev_index] = (228, 197, 85)

        if draw_result and draw_label:
            det_boxes3d, det_scores, det_cls, det_trackids = get_boxes_from_results(frame_results)
            gt_boxes3d, gt_scores, gt_cls, gt_trackids = get_boxes_from_labels(frame_labels)
            bev_map = draw_bev_boxes(bev_map=bev_map.copy(), boxes3d=gt_boxes3d, labels=gt_cls,
                                     scores=gt_scores, track_ids=gt_trackids,
                                     thickness = 2, colorize_with_label = False, color=(178, 255, 102),
                                     area_scope=area_scope)
            bev_map = draw_bev_boxes(bev_map = bev_map.copy(), boxes3d = det_boxes3d, labels = det_cls,
                                     scores = det_scores, track_ids = det_trackids,
                                     thickness = 2, colorize_with_label = False, color=(45, 151, 255),
                                     area_scope = area_scope)
        else:
            if draw_result:
                det_boxes3d, det_scores, det_cls, det_trackids = get_boxes_from_results(frame_results)
                bev_map = draw_bev_boxes(bev_map = bev_map.copy(), boxes3d = det_boxes3d, labels = det_cls,
                                         scores = det_scores, track_ids = det_trackids,
                                         area_scope = area_scope)
            if draw_label:
                gt_boxes3d, gt_scores, gt_cls, gt_trackids = get_boxes_from_labels(frame_labels)
                bev_map = draw_bev_boxes(bev_map = bev_map.copy(), boxes3d = gt_boxes3d, labels = gt_cls,
                                         scores = gt_scores, track_ids = gt_trackids,
                                         area_scope = area_scope)
            if draw_poly:
                poly_boxes3d, poly_vertices3d, poly_cls, poly_trackids = get_polys(frame_polys)
                bev_map = draw_bev_polys(bev_map = bev_map.copy(), polys = poly_vertices3d)
        bev_image = bev_map.astype(np.uint8)
        if draw_voxel:
            bev_image = visualize_utils.draw_bev_voxels(bev_image, voxel_path = frame_voxel_path,
                                                        voxel_size = voxel_size, area_scope = area_scope)
        if draw_image:
            bev_image = visualize_utils.draw_image(bev_image, image_path = frame_image_path, image_ratio = 0.2)
        return bev_image


def bev_visualization(draw_frame, draw_sequence, args, cloud_list, draw_cloud = True,
                      draw_result = False, draw_label = False, draw_poly = False, draw_voxel = False, draw_image = False,
                      results = None, labels = None, polys = None, voxel_list = None, image_list = None,
                      colorize_by_intensity = False):
    if draw_frame:
        frame_cloud_path = cloud_list[0]
        frame_results, frame_labels, frame_polys = None, None, None
        frame_voxel_path, frame_image_path = None, None
        if draw_result:
            frame_results = results[0]
        if draw_label:
            frame_labels = labels[0]
        if draw_poly:
            frame_polys = polys[0]
        if draw_voxel:
            frame_voxel_path = voxel_list[0]
        if draw_image:
            frame_image_path = image_list[0]

        frame_bev_map = draw_bev_map(frame_cloud_path, frame_results, frame_labels, frame_polys,
                                     frame_voxel_path, frame_image_path, draw_cloud, draw_result,
                                     draw_label, draw_poly, draw_voxel, draw_image,
                                     args.with_ring, area_scope, colorize_by_intensity)
        if args.draw_scale:
            frame_bev_map = visualize_utils.draw_bev_circle_scale(
                frame_bev_map, voxel_size=voxel_size, area_scope=area_scope, color = (200, 200, 200)
            )
        frame_bev_path = (frame_cloud_path.split('/')[-1]).split('.')[0] + '.png'
        if os.path.isdir(args.visual):
            # If the directory is provided, the map is writed into the directory
            frame_bev_fn = (frame_cloud_path.split('/')[-1]).rsplit('.', 1)[0] + '.png'
            frame_bev_path = os.path.join(args.visual, frame_bev_fn)
            print('Visualized Frame Path: ', frame_bev_path)
        elif args.visual.endswith('.png'):
            # If the png path is provided, the map is save as the path
            frame_bev_path = args.visual
            print('Visualized Frame Path: ', frame_bev_path)
        if not cv2.imwrite(frame_bev_path, frame_bev_map):
            warnings.warn('Write Image Error! Please check the path!')
            return False

    elif draw_sequence:
        if not os.path.exists(args.visual):
            os.makedirs(args.visual)
        print('Visualized Sequence Path: ', args.visual)
        for i in tqdm(range(len(cloud_list))):
            if i < args.beg_index or i >= args.end_index:
                continue
            frame_cloud_path = os.path.join(args.cloud, cloud_list[i])
            frame_results, frame_labels, frame_polys = None, None, None
            frame_voxel_path, frame_image_path = None, None
            if draw_result:
                frame_results = results[i]
            if draw_label:
                frame_labels = labels[i]
            if draw_poly:
                frame_polys = polys[i]
            if draw_voxel:
                frame_voxel_path = os.path.join(args.voxel, voxel_list[i])
            if draw_image:
                frame_image_path = os.path.join(args.image, image_list[i])

            frame_bev_map = draw_bev_map(frame_cloud_path, frame_results, frame_labels, frame_polys,
                                         frame_voxel_path, frame_image_path, draw_cloud, draw_result,
                                         draw_label, draw_poly, draw_voxel, draw_image,
                                         args.with_ring, area_scope, colorize_by_intensity)
            if args.draw_scale:
                frame_bev_map = visualize_utils.draw_bev_circle_scale(
                    frame_bev_map, voxel_size=voxel_size, area_scope=area_scope, color=(200, 200, 200)
                )
            frame_bev_fn = os.path.splitext(cloud_list[i])[0] + '.png'
            if not cv2.imwrite(os.path.join(args.visual, frame_bev_fn), frame_bev_map):
                warnings.warn('Write Image Error! Please check the path!')
                return False
    return True

def poly_trans(coor):
    coors=[]
    for i in range(0,len(coor),3):
        coors.append([coor[i],coor[i+1],coor[i+2]])
    return np.array(coors)


def scene_visualization(draw_frame, draw_sequence, args, cloud_list, draw_cloud = True,
                        draw_result = False, draw_label = False, draw_poly = False, draw_voxel = False, draw_image = False,
                        results = None, labels = None, polys = None, voxel_list = None, image_list = None):
    import mayavi.mlab as mlab
    # mlab.options.offscreen = True
    if draw_frame:
        frame_cloud_path = cloud_list[0]
        frame_results, frame_labels, frame_polys = None, None, None
        frame_voxel_path, frame_image_path = None, None
        if draw_result:
            frame_results = results[0]
        if draw_label:
            frame_labels = labels[0]
        if draw_poly:
            frame_polys = polys[0]
        if draw_voxel:
            frame_voxel_path = voxel_list[0]
        if draw_image:
            frame_image_path = image_list[0]

        frame_3d_fig = draw_3d_map(frame_cloud_path, frame_results, frame_labels, frame_polys, frame_voxel_path, frame_image_path,
            draw_cloud = draw_cloud, draw_result = draw_result, draw_label = draw_label, draw_poly = draw_poly,
            draw_voxel = draw_voxel, draw_image = draw_image, draw_ground = args.draw_ground,
            cloud_with_ring = args.with_ring, v2x = args.v2x, use_screenshot = args.demo, debug_mode = args.debug)

        frame_3d_path = (frame_cloud_path.split('/')[-1]).split('.')[0] + '.png'
        if os.path.isdir(args.visual):
            # If the directory is provided, the map is writed into the directory
            frame_3d_fn = (frame_cloud_path.split('/')[-1]).split('.')[0] + '.png'
            frame_3d_path = os.path.join(args.visual, frame_3d_fn)
        elif args.visual.endswith('.png'):
            # If the png path is provided, the map is save as the path
            frame_3d_path = args.visual
        if not (args.demo or draw_image):
            mlab.savefig(frame_3d_path)
            mlab.close()
        else:
            if not cv2.imwrite(frame_3d_path, frame_3d_fig):
                warnings.warn('Write Image Error! Please check the path!')
                return False

    elif draw_sequence:
        if not os.path.exists(args.visual):
            os.makedirs(args.visual)
        for i in tqdm(range(len(cloud_list))):
            # Valid index: beg_index < i < end_index
            if i < args.beg_index or i >= args.end_index:
                continue
            frame_cloud_path = os.path.join(args.cloud, cloud_list[i])
            frame_results, frame_labels, frame_polys = None, None, None
            frame_voxel_path, frame_image_path = None, None
            if draw_result:
                frame_results = results[i]
            if draw_label:
                frame_labels = labels[i]
            if draw_poly:
                frame_polys = polys[i]
            if draw_voxel:
                frame_voxel_path = os.path.join(args.voxel, voxel_list[i])
            if draw_image:
                frame_image_path = os.path.join(args.image, image_list[i])

            frame_3d_fig = draw_3d_map(frame_cloud_path, frame_results, frame_labels, frame_polys, frame_voxel_path, frame_image_path,
                draw_cloud=draw_cloud, draw_result=draw_result, draw_label=draw_label, draw_poly=draw_poly,
                draw_voxel=draw_voxel, draw_image = draw_image, draw_ground=args.draw_ground,
                cloud_with_ring=args.with_ring, v2x=args.v2x, use_screenshot=args.demo)

            frame_3d_fn = os.path.splitext(cloud_list[i])[0] + '.png'
            if not (args.demo or draw_image):
                mlab.savefig(os.path.join(args.visual, frame_3d_fn))
                mlab.close()
            else:
                if not cv2.imwrite(os.path.join(args.visual, frame_3d_fn), frame_3d_fig):
                    warnings.warn('Write Image Error! Please check the path!')
                    return False
                else:
                    sleep(0.05)
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cloud Visualizer')
    parser.add_argument('-c', '--cloud', type=str, default=None, help='Cloud path (file or directory)')
    parser.add_argument('-i', '--image', type=str, default=None, help='Image path (file or directory)')
    parser.add_argument('-l', '--label', type=str, default=None, help='Label path (file or directory)')
    parser.add_argument('-r', '--result', type=str, default=None, help='Result path (file or directory)')
    parser.add_argument('-p', '--poly', type=str, default=None, help='Polygon result path (file or directory)')
    parser.add_argument('-v', '--voxel', type=str, default=None, help='Voxel map path (file or directory)')

    parser.add_argument('--visual', type=str, default='visual_image', help='Save path for BEV or 3D maps (file or directory)')
    parser.add_argument('--v2x', action='store_true', default=False, help='Set the V2X view')
    parser.add_argument('--draw_3d', action='store_true', default=False, help='Draw 3D maps')
    parser.add_argument('--draw_scale', action='store_true', default=False, help='Draw the circle scales on BEV map')
    parser.add_argument('--draw_ground', action='store_true', default=False, help='Draw ground plane on 3D map')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--ground_height', type=float, default=-1.60, help='Ground height')
    parser.add_argument('--demo', action='store_true', default=False, help='Use screenshot to save 3D maps and make demo')
    parser.add_argument('--with_ring', action='store_true', default=False, help='Cloud with ring')
    parser.add_argument('--sort_by_num', action='store_true', default=False, help='Sort the files by number instead of name')
    parser.add_argument('--beg_index', type=int, default=0, help='Begin index for sequence')
    parser.add_argument('--end_index', type=int, default=10000, help='End index for sequence')
    parser.add_argument('--video_path', type=str, default=None, help='Video path')
    parser.add_argument('--colorize_by_intensity', action='store_true', default=False, help='Colorize the BEV cloud by intensity')

    args = parser.parse_args()
    # Check the inputs
    draw_frame, draw_sequence, draw_cloud, draw_result, draw_label, draw_poly, draw_voxel, draw_image = \
        check_args(args.cloud, args.result, args.label, args.poly, args.voxel, args.image)
    # Set the area scope and detected scope
    # For car
    area_scope = [[-72, 92], [-72, 72], [-5, 5]]
    det_scope = [[-40, 80], [-60, 60]]
    # For v2x
    if args.v2x:
        area_scope = [[-20, 80], [-80, 80], [-5, 5]]
        det_scope = [[-60, 60], [-60, 60]]

    # For Demo
    if args.demo:
        # Update the color for pedestrian
        colormap = [
            [0, 0, 0],
            [68, 255, 117],     # Car: Green
            [255, 151, 45],     # Pedestrian: Dark Orange
            [255, 204, 45],     # Cyclist: Gold Orange
            [142, 118, 255],    # Truck: Purple
            [238, 238, 0],      # Cone: Yellow
            [224, 224, 224],    # Unknown: Light Grey
            [190, 190, 190]     # DontCare: Grey
        ]
    # Load the results and labels
    cloud_list, voxel_list, image_list = [], [], []
    results, labels, polys, voxels = [], [], [], []
    if draw_frame:
        if draw_cloud:
            cloud_list.append(args.cloud)
        if draw_result:
            results = load_single_result(args.result)
        if draw_label:
            labels = load_single_label(args.label)
        if draw_poly:
            polys = load_single_poly(args.poly)
        if draw_voxel:
            voxel_list.append(args.voxel)
        if draw_image:
            image_list.append(args.image)
    elif draw_sequence:
        if draw_cloud:
            cloud_list = get_cloud_list(args.cloud, args.sort_by_num)
        if draw_result:
            results = load_results(args.result, args.sort_by_num)
        if draw_label:
            labels = load_labels(args.label, args.sort_by_num)
        if draw_poly:
            polys = load_polys(args.poly, args.sort_by_num)
        if draw_voxel:
            voxel_list = get_voxel_list(args.voxel, args.sort_by_num)
        if draw_image:
            image_list = get_image_list(args.image, args.sort_by_num)

    # Draw BEV maps or 3D maps
    if not args.draw_3d:
        # Draw BEV maps
        bev_visualization(draw_frame=draw_frame, draw_sequence=draw_sequence, args=args, draw_cloud=draw_cloud,
            draw_result=draw_result, draw_label=draw_label, draw_poly=draw_poly, draw_voxel = draw_voxel, draw_image = draw_image,
            cloud_list=cloud_list, results=results, labels=labels, polys = polys, voxel_list = voxel_list, image_list = image_list,
            colorize_by_intensity = args.colorize_by_intensity)
    else:
        # Draw 3D maps
        scene_visualization(draw_frame = draw_frame, draw_sequence = draw_sequence, args = args, draw_cloud=draw_cloud,
            draw_result=draw_result, draw_label=draw_label, draw_poly=draw_poly, draw_voxel = draw_voxel, draw_image = draw_image,
            cloud_list = cloud_list, results = results, labels = labels, polys = polys, voxel_list = voxel_list, image_list = image_list)

    # Write the images to video
    if draw_sequence and args.video_path is not None:
        # Create the video directory
        if not os.path.exists(os.path.dirname(args.video_path)):
            os.makedirs(os.path.dirname(args.video_path))
        if not args.draw_3d:
            image_list = os.listdir(args.visual)
        else:
            image_list = os.listdir(args.visual)
        if args.sort_by_num:
            image_list = sorted(image_list, key=lambda x: int(x.split('.')[0]))
        else:
            image_list = sorted(image_list)

        # Initialize the video object
        cloud_image = cv2.imread(os.path.join(args.visual, image_list[0]))
        frame_height, frame_width, frame_channels = cloud_image.shape

        # Check the video path
        video_dir = os.path.dirname(args.video_path)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        if not args.video_path.endswith('.avi'):
            args.video_path = args.video_path.split('.')[0] + '.avi'
        cloud_video = cv2.VideoWriter(args.video_path, cv2.VideoWriter_fourcc(*'MPEG'), 10,
                                      (frame_width,frame_height))
        for index, image_fn in enumerate(image_list):
            if index < args.beg_index or index >= args.end_index:
                continue
            cloud_video.write(cv2.imread(os.path.join(args.visual, image_fn)))
        cv2.destroyAllWindows()
        cloud_video.release()
