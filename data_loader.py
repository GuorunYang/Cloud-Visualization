import os
import json
import numpy as np
import pickle5 as pkl
from tqdm import tqdm
from pypcd import pypcd


classes = [
    'Car',
    'Pedestrian',
    'Cyclist',
    'Truck',
    'Misc',
    'Cone',
    'Unknown',
    # 'Dontcare',
    # 'Traffic_Warning_Object',
    # 'Traffic_Warning_Sign',
    # 'Road_Falling_Object',
    # 'Road_Intrusion_Object',
    # 'Animal'
]
cls_dict = {
    'smallmot'      : 'Vehicle',
    'bigmot'        : 'Vehicle',
    'trafficcone'   : 'Misc',
    'pedestrian'    : 'Pedestrian',
    'crashbarrel'   : 'Misc',
    'tricyclist'    : 'Cyclist',
    'bicyclist'     : 'Cyclist',
    'motorcyclist'  : 'Cyclist',
    'onlybicycle'   : 'Cyclist',
    'crowd'         : 'Misc',
    'onlytricycle'  : 'Cyclist',
    'stopbar'       : 'Unknown',
    'smallmovable'  : 'Misc',
    'safetybarrier' : 'Unknown',
    'smallunmovable': 'Misc',
    'warningpost'   : 'Misc',
    'fog'           : 'Unknown',
    'sign'          : 'Misc',
}


def load_cloud(cloud_path):
	if cloud_path.endswith('.bin'):
	    return np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 5)
	elif cloud_path.endswith('.pcd'):
		pcd_cloud = pypcd.PointCloud.from_path(cloud_path)
		points = np.zeros([pcd_cloud.width, 4], dtype=np.float32)
		points[:, 0] = pcd_cloud.pc_data['x'].copy()
		points[:, 1] = pcd_cloud.pc_data['y'].copy()
		points[:, 2] = pcd_cloud.pc_data['z'].copy()
		points[:, 3] = pcd_cloud.pc_data['intensity'].copy()
		return points
	elif cloud_path.endswith('.txt'):
	    return np.loadtxt(cloud_path, dtype = np.float32)
	elif cloud_path.endswith('.npy'):
		return np.load(cloud_path, dtype = np.float32)
	else:
	    raise TypeError('Cannot support cloud format')

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

def load_pkl_label(pkl_label_path, query_frame = None, sort_by_num = False):
    annotation_cls = list(cls_dict.keys())
    with open(pkl_label_path, 'rb') as f:
        pkl_labels = pkl.load(f)
        max_frames = len(pkl_labels)
        labels = [{'annos': {}} for i in range(max_frames)]
        # Scan each frame
        frame_id = 0
        query_flag = False
        for frame_name, frame_label in tqdm(pkl_labels.items()):
            if not query_frame is None:
                if not query_frame in frame_name:
                    continue
                else:
                    query_flag = True
                    print("Query frame {} is in label pkl".format(query_frame))
            for i, anno in enumerate(frame_label):
                anno_type = cls_dict[anno['original_lbl'].lower()]
                # anno_dimensions = [anno['height'], anno['width'], anno['length']]
                anno_dimensions = [anno['width'], anno['length'], anno['height']]
                anno_locations = [anno['x'], anno['y'], anno['z']]
                anno_rotation_y = [np.pi / 2.0 - anno['rotation_y'] + np.pi]
                anno_box3d = anno_locations + anno_dimensions + anno_rotation_y
                if anno_type == 'Vehicle':
                    if anno['length'] >= 6.0:
                        anno_type = 'Truck'
                    else:
                        anno_type = 'Car'
                if 'gt_box' in labels[frame_id]['annos']:
                    labels[frame_id]['annos']['gt_box'].append(anno_box3d)
                else:
                    labels[frame_id]['annos']['gt_box'] = [anno_box3d]
                if 'name' in labels[frame_id]['annos']:
                    labels[frame_id]['annos']['name'].append(anno_type)
                else:
                    labels[frame_id]['annos']['name'] = [anno_type]
            frame_id += 1
        for i in range(len(labels)):
            for key in labels[i]['annos']:
                labels[i]['annos'][key] = np.array(labels[i]['annos'][key])
        if not query_frame is None:
            if not query_flag:
                print("Query frame NOT in label pkl".format(query_frame))
        return labels

def load_dir_label(label_dir, query_frame = None, sort_by_num = False):
    label_list = os.listdir(label_dir)
    if sort_by_num:
        label_list = sorted(label_list, key=lambda x: int(x.split('.')[0]))
    else:
        label_list = sorted(label_list)
    max_frames = len(label_list)
    labels = [{'annos': {}} for i in range(max_frames)]
    query_flag = False
    for frame_id, label_fn in enumerate(label_list):
        label_path = os.path.join(label_dir, label_fn)
        if not query_frame is None:
            if not query_frame in label_fn:
                continue
            else:
                query_flag = True
                print("Query frame {} is in label directory".format(query_frame))
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
    if not query_frame is None:
        if not query_flag:
            print("Query frame NOT in label directory".format(query_frame))
    return labels


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

def load_labels(label_pth, query_frame = None, sort_by_num = False):
    if os.path.isdir(label_pth):
        labels = load_dir_label(label_pth, query_frame)
        return labels
    elif os.path.isfile(label_pth):
        if label_pth.endswith('.pkl'):
            labels = load_pkl_label(label_pth, query_frame)
            return labels
        elif label_pth.endswith('.txt'):
            labels = load_single_label(label_pth)
            return labels
    else:
        raise TypeError('Cannot load labels')
        return None



def get_cloud_list(cloud_dir, sort_by_num = False):
    cloud_list = os.listdir(cloud_dir)
    if sort_by_num:
        cloud_list = sorted(cloud_list, key=lambda x: int(x.split('.')[0]))
    else:
        cloud_list = sorted(cloud_list)
    cloud_list = [os.path.join(cloud_dir, cloud_name) for cloud_name in cloud_list]
    return cloud_list

def get_voxel_list(voxel_dir, sort_by_num = False):
    voxel_list = os.listdir(voxel_dir)
    if sort_by_num:
        voxel_list = sorted(voxel_list, key=lambda x: int(x.split('.')[0]))
    else:
        voxel_list = sorted(voxel_list)
    voxel_list = [os.path.join(voxel_dir, voxel_name) for voxel_name in voxel_list]
    return voxel_list

def get_image_list(image_dir, sort_by_num = False):
    image_list = os.listdir(image_dir)
    if sort_by_num:
        image_list = sorted(image_list, key=lambda x: int(x.split('.')[0]))
    else:
        image_list = sorted(image_list)
    image_list = [os.path.join(image_dir, image_name) for image_name in image_list]
    return image_list

def get_polys(frame_polys = None):
    if frame_polys is not None:
        # classes = ['', 'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Cone', 'Unknown', 'Dontcare']
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
        # classes = ['', 'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Cone', 'Unknown', 'Dontcare',
        #            'Traffic_Warning_Object', 'Traffic_Warning_Sign',
        #            'Road_Falling_Object', 'Road_Intrusion_Object', 'Animal'
        #            ]
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
        # classes = ['', 'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Cone', 'Unknown', 'Dontcare',
        #            'Traffic_Warning_Object', 'Traffic_Warning_Sign',
        #            'Road_Falling_Object', 'Road_Intrusion_Object', 'Animal'
        #           ]
        gt_boxes3d = None if 'gt_box' not in frame_labels['annos'] else frame_labels['annos']['gt_box']
        gt_types = None if 'name' not in frame_labels['annos'] else  frame_labels['annos']['name']
        gt_scores = None if 'score' not in frame_labels['annos'] else frame_labels['annos']['score']
        gt_trackids = None if 'track_info' not in frame_labels['annos'] else frame_labels['annos']['track_info']
        gt_labels = None if gt_types is None else np.array([classes.index(type) for type in gt_types])
        return gt_boxes3d, gt_scores, gt_labels, gt_trackids
    else:
        return None



def count_number(frame_results = None, frame_labels = None, view_region = [0, 40, -50, 50]):
    count_dict = {
        'Vehicle': 0, 'Pedestrian' : 1, 'Cyclist' : 2, 'Misc' : 2
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