import os
import json
import numpy as np
import pickle5 as pkl
from tqdm import tqdm
from pypcd import pypcd
import rclone

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
    'mpv'           : 'Vehicle',
    'car'           : 'Vehicle',
    'bigboxtruck'   : 'Vehicle',
    'smallboxtruck' : 'Vehicle',
    'suv'           : 'Vehicle',
    'truck'         : 'Vehicle',
    'pickupcar'     : 'Vehicle',
    'bus'           : 'Vehicle',
    'ghost'         : 'Vehicle',
    'crashbarrier'  : 'Misc',
    'person'        : 'Pedestrian',
    'warningtriangle'       : 'Misc',
    'engineeringvehicle'    : 'Vehicle',
    'unknownmovable'        : 'Vehicle',
    'othercar'              : 'Vehicle',
    'tralier'               : 'Vehicle',
    'ignore'                : 'Unknown',
    'motor'                 : 'Cyclist',
    'tricycle'              : 'Cyclist',
    'door'                  : 'Vehicle',
    'bike'                  : 'Cyclist',
    'unknownunmovable'      : 'Misc',
    'boxtypetricyclist'     : 'Cyclist',
    'boxtypetricycle'       : 'Cyclist',
    'trolley'               : 'Cyclist',
    'triangleroadblock'     : 'Misc',
}
det_cls_dict = {
    1 : 'Vehicle',
    2 : 'Pedestrian',
    3 : 'Cyclist', 
    4 : 'Misc',
}
check_cls_dict = {
    0 : 'Vehicle',
    1 : 'Pedestrian',
    2 : 'Cyclist', 
    3 : 'Misc',    
}

json_cls_dict = {
    # 'TYPE_UNKNOWN' : 'Unknown', 
    'VEHICLE'       : 'Vehicle', 
    'PEDESTRIAN'    : 'Pedestrian',
    'CYCLIST'       : 'Cyclist',
    'MISC'          : 'Misc'
}

colormap = [
    # [0, 0, 0],
    [68, 255, 117],     # 0 Car: Green
    [255, 51, 51],      # 1 Pedestrian: Red
    [255, 204, 45],     # 2 Cyclist: Gold Orange
    [142, 118, 255],    # 3 Truck: Purple
    [224, 224, 224],    # 4 Misc: Light Grey
    [224, 224, 224],    # 5 Unknown: Light Grey
    [190, 190, 190],    # 6 DontCare: Grey
    [255, 215, 0],      # 7 Traffic_Warning_Object: Gold
    [255, 192, 203],    # 8 Traffic_Warning_Sign: Pink
    [255, 127, 36],    # 9 Road_Falling_Object: Chocolate1
    [255, 64, 64],    # 10 Road_Intrusion_Object: Brown1
    [255, 0, 255],    # 11 Animal: Magenta
]

# colormap = [
#     [0, 0, 0],
#     [68, 255, 117],     # 0 Car: Green
#      # [255, 151, 45],  # 1 Pedestrian: Dark Orange
#     [255, 51, 51],      # 1 Pedestrian: Red
#     [255, 204, 45],     # 2 Cyclist: Gold Orange
#     [142, 118, 255],    # 3 Truck: Purple
#     [224, 224, 224],    # 4 Misc: Light Grey
#     [224, 224, 224],    # 5 Unknown: Light Grey
#     [190, 190, 190],    # 6 DontCare: Grey
#     [255, 215, 0],      # 7 Traffic_Warning_Object: Gold
#     [255, 192, 203],    # 8 Traffic_Warning_Sign: Pink
#     [255, 127, 36],     # 9 Road_Falling_Object: Chocolate1
#     [255, 64, 64],      # 10 Road_Intrusion_Object: Brown1
#     [255, 0, 255],      # 11 Animal: Magenta
# ]


def check_local_pth(file_path):
    return os.path.exists(file_path)

def check_file_path(file_path):
    if file_path.startswith("tos://"):
        with open(".rclone.conf") as f:
            cfg = f.read()
            result = rclone.with_config(cfg).ls(file_path)
            if (result.get('code') == 0) and (result.get('out') == b''):
                return False
            else:
                return True
    else:
        return os.path.exists(file_path)


def download_remote_file(remote_path, local_path = None):
    if local_path is None:
        local_path = remote_path.split("/")[-1]
    with open(".rclone.conf") as f:
        cfg = f.read()
        rclone.with_config(cfg).copy(remote_path, local_path)


def load_cloud(cloud_path, rotation = 0.0):
    points = np.zeros((0, 4), dtype=np.float32)
    if cloud_path.endswith('.bin'):
        points = np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 4)
    elif cloud_path.endswith('.pcd'):
        pcd_cloud = pypcd.PointCloud.from_path(cloud_path)
        points = np.zeros([pcd_cloud.width, 4], dtype=np.float32)
        points[:, 0] = pcd_cloud.pc_data['x'].copy()
        points[:, 1] = pcd_cloud.pc_data['y'].copy()
        points[:, 2] = pcd_cloud.pc_data['z'].copy()
        points[:, 3] = pcd_cloud.pc_data['intensity'].copy()
    elif cloud_path.endswith('.txt'):
        points = np.loadtxt(cloud_path, dtype = np.float32)
    elif cloud_path.endswith('.npy'):
        points = np.load(cloud_path).astype(np.float32)
    else:
        raise TypeError('Cannot support cloud format')
    if points.shape[0] > 0:
        points_xyz = points[:, :3]
        points = points[~(points_xyz == 0).all(1)]  # Remove zero points
        points = points[~np.isnan(points).any(axis=1)]  # Remove NaN points

        if rotation != 0.0:
            # Convert the angle from degrees to radians
            points_xyz = points[:, :3]
            angle_rad = np.radians(rotation)
            # Define the rotation matrix around the Z-axis
            rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                        [np.sin(angle_rad), np.cos(angle_rad), 0],
                                        [0, 0, 1]])
            # Apply the rotation to the point cloud
            rotated_points_xyz = np.dot(points_xyz, rotation_matrix.T)
            points[:, :3] = rotated_points_xyz
    return points


def save_cloud(des_cloud_path, cloud):
    '''
        cloud: nx4 array
    '''
    if cloud.shape[1] != 4:
        raise BaseException('The cols of cloud != 4')
        return False
    if des_cloud_path.endswith('.pcd'):
        des_point_type = np.dtype(
            [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.uint8)]
        )
        points = np.zeros(cloud.shape[0], dtype=des_point_type)
        points['x'] = cloud[:, 0]
        points['y'] = cloud[:, 1]
        points['z'] = cloud[:, 2]
        points['intensity'] = cloud[:, 3].astype(np.uint8)
        # Write the header
        with open(des_cloud_path, 'w') as fp:
            fp.write(
                '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 1\nTYPE F F F I\nCOUNT 1 1 1 1')
            fp.write('\nWIDTH ' + str(cloud.shape[0]))
            fp.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
            fp.write('\nPOINTS ' + str(cloud.shape[0]))
            fp.write('\nDATA binary')
            fp.write('\n')
        # Write the points
        with open(des_cloud_path, 'ab+') as fp:
            pc_data = np.array(points, dtype=des_point_type)
            fp.write(pc_data.tostring('C'))
        return True
    elif des_cloud_path.endswith('.bin'):
        cloud = cloud.astype(np.float32)
        cloud.tofile(des_cloud_path)
        return True
    else:
        raise TypeError('The suffix of cloud pth is not supported')
        return False


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
        dimensions = [float(line_array[6]), float(line_array[4]), float(line_array[5])]
        # dimensions = [float(line_array[5]), float(line_array[4]), float(line_array[6])]
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
        if line_array[1] in cls_dict:
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
        if line_array[1] in cls_dict:
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
    if det_path.endswith('.txt'):
        return load_single_txt(det_path)
    elif det_path.endswith('.json'):
        return load_single_json(det_path)
    else:
        print("Cannot support the format of det path {}".format(det_path))
    

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


def load_dir_result(det_dir, query_frame = None, sort_by_num = False):
    result_dict = {}
    det_list = os.listdir(det_dir)
    if sort_by_num:
        det_list = sorted(det_list, key=lambda x: int(x.split('.')[0]))
    else:
        det_list = sorted(det_list)

    query_flag = False
    for frame_id, det_fn in enumerate(det_list):
        frame_name  = os.path.splitext(os.path.basename(det_fn))[0] # Split the extension of frame name
        det_path = os.path.join(det_dir, det_fn)
        if not query_frame is None:
            if not query_frame in det_fn:
                continue
            else:
                query_flag = True
                print("Query frame {} is in result directory".format(query_frame))
        result_dict[frame_name] = {'dets': {}}
        if det_fn.endswith('.txt'):
            result_dict.update(load_single_txt(det_path))
        elif det_fn.endswith('.json'):
            frame_result = load_single_json(det_path)
            result_dict.update(load_single_json(det_path))
 
    for frame_name, frame_det in result_dict.items():
        for key in frame_det['dets']:
            frame_det['dets'][key] = np.array(frame_det['dets'][key])
    if not query_frame is None:
        if not query_flag:
            print("Query frame NOT in result pkl".format(query_frame))
    return result_dict


def load_pkl_result(pkl_result_path, query_frame = None, sort_by_num = False):
    result_dict = {}
    query_flag = False
    if query_frame is not None:
        query_frame, _ = os.path.splitext(query_frame)
    with open(pkl_result_path, 'rb') as f:
        pkl_results = pkl.load(f)
        for frame_name, frame_result in pkl_results.items():
            if "/" in frame_name:
                frame_name, _ = os.path.splitext(frame_name.split("/")[-1])
            if not query_frame is None:
                if not query_frame in frame_name:
                    continue
                else:
                    query_flag = True
                    print("Query frame {} is in label pkl".format(query_frame))

            result_dict[frame_name] = {
                "dets" : {
                    "name" : [], 
                    "det_box" : [], 
                    "score" : [],
                },
            }
            
            for i in range(len(frame_result['lbl'])):
                det_type = check_cls_dict[frame_result['lbl'][i]]
                det_length = frame_result['bbox'][i, 3]
                if det_type == 'Vehicle':
                    if det_length >= 6.0:
                        det_type = 'Truck'
                    else:
                        det_type = 'Car'
                result_dict[frame_name]['dets']['name'].append(det_type)
            result_dict[frame_name]['dets']['det_box'] = frame_result['bbox']
            result_dict[frame_name]['dets']['score'] = frame_result['score']
            rotation_y = np.pi / 2.0 - frame_result['bbox'][:, 6] + np.pi
            result_dict[frame_name]['dets']['det_box'][:, 6] = rotation_y
            result_dict[frame_name]["dets"]["det_box"][:, [3, 4]] = result_dict[frame_name]["dets"]["det_box"][:, [4, 3]]

    return result_dict

# def load_pkl_result(pkl_result_path, query_frame = None, sort_by_num = False):
#     result_dict = {}
#     query_flag = False

#     with open(pkl_result_path, 'rb') as f:
#         pkl_results = pkl.load(f)
#         for dataset_name, dataset_results in pkl_results.items():            
#             for frame_name, frame_result in dataset_results.items():
#                 if not query_frame is None:
#                     if not query_frame in frame_name:
#                         continue
#                     else:
#                         query_flag = True
#                         print("Query frame {} is in result pkl".format(query_frame))
#                 result_dict[frame_name] = {'dets': {}}
#                 for i, det_result in enumerate(frame_result):
#                     det_type = det_cls_dict[det_result['lbl']]
#                     det_locations = [float(det_result['x']), float(det_result['y']), float(det_result['z'])]
#                     det_dimensions = [float(det_result['width']), float(det_result['length']), float(det_result['height'])]
#                     # det_rotation_y = [float(det_result['rotation_y'])]
#                     det_rotation_y = [np.pi / 2.0 - det_result['rotation_y'] + np.pi]
#                     det_score = float(det_result['prob'])
#                     det_box = det_locations + det_dimensions + det_rotation_y
#                     if det_type == 'Vehicle':
#                         if det_result['length'] >= 6.0:
#                             det_type = 'Truck'
#                         else:
#                             det_type = 'Car'

#                     if 'det_box' in result_dict[frame_name]['dets']:
#                         result_dict[frame_name]['dets']['det_box'].append(det_box)
#                     else:
#                         result_dict[frame_name]['dets']['det_box'] = [det_box]
#                     if 'name' in result_dict[frame_name]['dets']:
#                         result_dict[frame_name]['dets']['name'].append(det_type)
#                     else:
#                         result_dict[frame_name]['dets']['name'] = [det_type]
#                     if 'score' in result_dict[frame_name]['dets']:
#                         result_dict[frame_name]['dets']['score'].append(det_score)
#                     else:
#                         result_dict[frame_name]['dets']['score'] = [det_score]

#     for frame_name, frame_det in result_dict.items():
#         for key in frame_det['dets']:
#             frame_det['dets'][key] = np.array(frame_det['dets'][key])
#     if not query_frame is None:
#         if not query_flag:
#             print("Query frame NOT in result pkl".format(query_frame))
#     return result_dict


def load_single_txt(det_path):
    '''
        Load detection results from single txt file
    '''
    result_dict = {}
    frame_name = os.path.splitext(det_path.split('/')[-1])[0]
    result_dict[frame_name] = {'dets': {}}
    with open(det_path, 'r') as f:
        det_lines = f.readlines()
        for det_ln in det_lines:
            det_type, det_box, det_score, det_iou, det_track = load_line(det_ln)
            if 'det_box' in result_dict[frame_name]['dets']:
                result_dict[frame_name]['dets']['det_box'].append(det_box)
            else:
                result_dict[frame_name]['dets']['det_box'] = [det_box]
            if 'name' in result_dict[frame_name]['dets']:
                result_dict[frame_name]['dets']['name'].append(det_type)
            else:
                result_dict[frame_name]['dets']['name'] = [det_type]
            if 'score' in result_dict[frame_name]['dets']:
                result_dict[frame_name]['dets']['score'].append(det_score)
            else:
                result_dict[frame_name]['dets']['score'] = [det_score]
            if det_track is not None:
                if 'track_info' in result_dict[frame_name]['dets']:
                    result_dict[frame_name]['dets']['track_info'].append(det_track)
                else:
                    result_dict[frame_name]['dets']['track_info'] = [det_track]
    for frame_name, frame_det in result_dict.items():
        for key in frame_det['dets']:
            frame_det['dets'][key] = np.array(frame_det['dets'][key])
    return result_dict


def load_single_json(det_path):
    '''
        Load detection results from single json file
    '''
    result_dict = {}
    frame_name = os.path.splitext(det_path.split('/')[-1])[0]
    result_dict[frame_name] = {'dets': {}}
    # results = [{'dets': {}} for i in range(1)]
    with open(det_path, 'r') as f:
        json_results = json.load(f)
        for i, det_obj in enumerate(json_results['obstacles']):
            det_type = json_cls_dict[det_obj['label_class']]
            det_score = det_obj['confidence']
            det_dimensions = [det_obj['width'], det_obj['length'], det_obj['height']]
            det_locations = [det_obj['center']['x'], det_obj['center']['y'], det_obj['center']['z']]
            det_rotation_y = [det_obj['orientation']]

            # hardcode for current detections
            det_locations = [det_obj['center']['y'], -det_obj['center']['x'], det_obj['center']['z']]
            det_rotation_y = [det_obj['orientation'] + np.pi]

            det_box = det_locations + det_dimensions + det_rotation_y
            if det_type == 'Vehicle':
                if det_obj['length'] >= 6.0:
                    det_type = 'Truck'
                else:
                    det_type = 'Car'
            if 'det_box' in result_dict[frame_name]['dets']:
                result_dict[frame_name]['dets']['det_box'].append(det_box)
            else:
                result_dict[frame_name]['dets']['det_box'] = [det_box]
            if 'name' in result_dict[frame_name]['dets']:
                result_dict[frame_name]['dets']['name'].append(det_type)
            else:
                result_dict[frame_name]['dets']['name'] = [det_type]
            if 'score' in result_dict[frame_name]['dets']:
                result_dict[frame_name]['dets']['score'].append(det_score)
            else:
                result_dict[frame_name]['dets']['score'] = [det_score]
    for frame_name, frame_det in result_dict.items():
        for key in frame_det['dets']:
            frame_det['dets'][key] = np.array(frame_det['dets'][key])
    return result_dict


def load_single_label(label_path):
    '''
        Load labels from single frame
    '''
    # labels = [{'annos': {}} for i in range(1)]
    frame_name = os.path.splitext(os.path.basename(label_path))[0]
    label_dict = {}
    label_dict[frame_name] = {"annos" : {}}
    with open(label_path, 'r') as f:
        label_lines = f.readlines()
        for label_ln in label_lines:
            label_type, label_box, label_score, label_iou, label_track = load_line(label_ln)

            # Append the annotation to label_dict
            if 'gt_box' in label_dict[frame_name]['annos']:
                label_dict[frame_name]['annos']['gt_box'].append(label_box)
            else:
                label_dict[frame_name]['annos']['gt_box'] = [label_box]
            if 'name' in label_dict[frame_name]['annos']:
                label_dict[frame_name]['annos']['name'].append(label_type)
            else:
                label_dict[frame_name]['annos']['name'] = [label_type]
            if label_track is not None:
                if 'track_info' in label_dict[frame_name]['annos']:
                    label_dict[frame_name]['annos']['track_info'].append(label_track)
                else:
                    label_dict[frame_name]['annos']['track_info'] = [label_track]

    for frame_name, frame_anno in label_dict.items():
        if 'annos' in frame_anno.keys():
            for key in frame_anno['annos']:
                frame_anno['annos'][key] = np.array(frame_anno['annos'][key])
    return label_dict


def load_pkl_check(pkl_check_path, query_frame = None, sort_by_num = False):
    check_dict = {}
    # Get the basename of query frame
    if query_frame is not None:
        query_frame, _ = os.path.splitext(query_frame)
    with open(pkl_check_path, 'rb') as f:
        pkl_checks = pkl.load(f)
        query_flag = False
        for frame_name, frame_check in tqdm(pkl_checks.items()):
            # Get the basename of frame
            if "/" in frame_name:
                frame_name, _ = os.path.splitext(frame_name.split("/")[-1])
            if not query_frame is None:
                if not query_frame in frame_name:
                    continue
                else:
                    query_flag = True
                    print("Query frame {} is in label pkl".format(query_frame))
            check_dict[frame_name] = {
                "annos" : {"name" : [], "gt_box" : []}, 
                "dets" : {"name" : [], "det_box" : []},
            }
            # Load the GT wrong
            for i in range(frame_check['gt_wrong_num']):
                check_type = check_cls_dict[frame_check['gt_wrong_label'][i]]
                obj_length = frame_check['gt_wrong_bbox'][i, 3]
                if check_type == 'Vehicle':
                    if obj_length >= 6.0:
                        check_type = 'Truck'
                    else:
                        check_type = 'Car'
                check_dict[frame_name]["annos"]["name"].append(check_type)
            check_dict[frame_name]["annos"]["name"] = np.array(check_dict[frame_name]["annos"]["name"])
            check_dict[frame_name]["annos"]["gt_box"] = frame_check['gt_wrong_bbox']
            check_dict[frame_name]["annos"]["gt_box"][:, [3, 4]] = check_dict[frame_name]["annos"]["gt_box"][:, [4, 3]]
            rotation_y = np.pi / 2.0 - frame_check['gt_wrong_bbox'][:, 6] + np.pi
            check_dict[frame_name]["annos"]["gt_box"][:, 6] = rotation_y

            # Load the GT miss
            for j in range(frame_check['gt_miss_num']):
                check_type = check_cls_dict[frame_check['gt_miss_label'][j]]
                obj_length = frame_check['gt_miss_bbox'][j, 3]
                if check_type == 'Vehicle':
                    if obj_length >= 6.0:
                        check_type = 'Truck'
                    else:
                        check_type = 'Car'
                check_dict[frame_name]["dets"]["name"].append(check_type)
            check_dict[frame_name]["dets"]["name"] = np.array(check_dict[frame_name]["dets"]["name"])
            check_dict[frame_name]["dets"]["det_box"] = frame_check['gt_miss_bbox']
            check_dict[frame_name]["dets"]["det_box"][:, [3, 4]] = check_dict[frame_name]["dets"]["det_box"][:, [4, 3]]
            rotation_y = np.pi / 2.0 - frame_check['gt_miss_bbox'][:, 6] + np.pi
            check_dict[frame_name]["dets"]["det_box"][:, 6] = rotation_y

        if not query_frame is None:
            if not query_flag:
                print("Query frame NOT in label pkl".format(query_frame))
    return check_dict
            

def load_pkl_label(pkl_label_path, query_frame = None, sort_by_num = False):
    label_dict = {}
    annotation_cls = list(cls_dict.keys())
    with open(pkl_label_path, 'rb') as f:
        pkl_labels = pkl.load(f)
        # max_frames = len(pkl_labels)
        # labels = [{'annos': {}} for i in range(max_frames)]

        # Scan each frame
        query_flag = False
        for frame_name, frame_label in tqdm(pkl_labels.items()):
            if not query_frame is None:
                if not query_frame in frame_name:
                    continue
                else:
                    query_flag = True
                    print("Query frame {} is in label pkl".format(query_frame))
            label_dict[frame_name] = {"annos" : {}}
            for i, anno in enumerate(frame_label):
                anno_type = ''
                try:
                    anno_type = cls_dict[anno['original_lbl'].lower()]
                except:
                    print("Origin label : {} , New label : {}".format(anno['original_lbl'], anno['lbl']))
                    continue
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
                # Append the annotation to label_dict
                if 'gt_box' in label_dict[frame_name]['annos']:
                    label_dict[frame_name]['annos']['gt_box'].append(anno_box3d)
                else:
                    label_dict[frame_name]['annos']['gt_box'] = [anno_box3d]
                if 'name' in label_dict[frame_name]['annos']:
                    label_dict[frame_name]['annos']['name'].append(anno_type)
                else:
                    label_dict[frame_name]['annos']['name'] = [anno_type]

        for frame_name, frame_anno in label_dict.items():
            if 'annos' in frame_anno.keys():
                for key in frame_anno['annos']:
                    frame_anno['annos'][key] = np.array(frame_anno['annos'][key])
        if not query_frame is None:
            if not query_flag:
                print("Query frame NOT in label pkl".format(query_frame))

        return label_dict


def load_dir_label(label_dir, query_frame = None, sort_by_num = False):
    label_dict = {}
    label_list = os.listdir(label_dir)
    if sort_by_num:
        label_list = sorted(label_list, key=lambda x: int(x.split('.')[0]))
    else:
        label_list = sorted(label_list)
    max_frames = len(label_list)
    # labels = [{'annos': {}} for i in range(max_frames)]

    query_flag = False
    for frame_id, label_fn in enumerate(label_list):
        frame_name  = os.path.splitext(os.path.basename(label_fn))[0] # Split the extension of frame name
        label_path = os.path.join(label_dir, label_fn)
        if not query_frame is None:
            if not query_frame in label_fn:
                continue
            else:
                query_flag = True
                print("Query frame {} is in label directory".format(query_frame))
        label_dict[frame_name] = {"annos" : {}}
        with open(label_path, 'r') as f:
            label_lines = f.readlines()
            for label_ln in label_lines:
                label_type, label_box, label_score, label_iou, label_track = load_line(label_ln)
                # Append the annotation to label_dict
                if 'gt_box' in label_dict[frame_name]['annos']:
                    label_dict[frame_name]['annos']['gt_box'].append(label_box)
                else:
                    label_dict[frame_name]['annos']['gt_box'] = [label_box]
                if 'name' in label_dict[frame_name]['annos']:
                    label_dict[frame_name]['annos']['name'].append(label_type)
                else:
                    label_dict[frame_name]['annos']['name'] = [label_type]
                if label_track is not None:
                    if 'track_info' in label_dict[frame_name]['annos']:
                        label_dict[frame_name]['annos']['track_info'].append(label_track)
                    else:
                        label_dict[frame_name]['annos']['track_info'] = [label_track]
    for frame_name, frame_anno in label_dict.items():
        if 'annos' in frame_anno.keys():
            for key in frame_anno['annos']:
                frame_anno['annos'][key] = np.array(frame_anno['annos'][key])
    if not query_frame is None:
        if not query_flag:
            print("Query frame NOT in label directory".format(query_frame))

    # print("Label Dict: ", label_dict)
    return label_dict


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


def load_results(result_pth, query_frame = None, sort_by_num = False):
    if os.path.isdir(result_pth):
        results = load_dir_result(result_pth, query_frame)
        return results
    elif os.path.isfile(result_pth):
        if result_pth.endswith('.pkl'):
            results = load_pkl_result(result_pth, query_frame)
            return results
        elif result_pth.endswith('.txt'):
            results = load_single_txt(result_pth)
            return results
        elif result_pth.endswith('.json'):
            results = load_single_json(result_pth)
            return results
        else:
            print("Cannot support result format")
        # elif result_pth.endswith('.json'):
        #     results = load_single_json(result_pth)
    else:
        raise TypeError('Cannot load labels')
        return None    


def load_checks(check_pth, query_frame = None, sort_by_num = False):
    if os.path.isfile(check_pth):
        if check_pth.endswith('.pkl'):
            checks = load_pkl_check(check_pth, query_frame)
            return checks
    else:
        raise TypeError('Cannot load checks')
        return None 


def load_labels(label_pth, query_frame = None, sort_by_num = False):
    print("Label pth: ", label_pth)
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
        return None, None, None, None


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
        return None, None, None, None


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
