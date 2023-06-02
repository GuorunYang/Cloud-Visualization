import os
import argparse
import warnings
import cv2
import numpy as np
from tqdm import tqdm
from time import sleep
import data_loader
from visual_bev import VisualBEV
from visual_3d import Visual3D

# score_thresh = [0, 0.6, 0.45, 0.5, 0.6, 0, 0, 0]       # For Model V0.7
# score_thresh = [0, 0.50, 0.45, 0.50, 0.50, 0, 0, 0]    # For Model V0.8
# score_thresh = [0, 0.64, 0.43, 0.55, 0.64]      # For Model V2.3
# score_thresh = [0, 0.58, 0.44, 0.53, 0.63]      # For Model V2.5
# score_thresh = [0, 0.62, 0.43, 0.55, 0.62]      # For Model V2.6

v2x_region = [0, 40, -50, 50]
def get_cls():
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
    return cls_dict, cls_array, colormap

def check_args(args):
    '''
        Check the input arguments and set the status
            1. draw frame or draw sequence
            2. draw cloud, draw result, draw label, draw poly, draw voxel, or draw image
    '''

    # Step 0: Define the Arguments
    draw_status = {
        "draw_frame"    : False,
        "draw_sequence" : False,
        "draw_cloud"    : False,
        "draw_result"   : False, 
        "draw_label"    : False,
        "draw_poly"    : False,
        "draw_voxel"    : False,
        "draw_image"    : False,
        "draw_scale"    : False, 
        "draw_intensity": False,
        "draw_ground"   : False,
        "use_screenshot" : False,
        "debug" : False,
    }
    cloud_format = {
        "bin_cloud" : False,
        "pcd_cloud" : False,
        "txt_cloud" : False,
        "npy_cloud" : False,
    }
    cloud_list = []
    # Step 1: Check the cloud
    if os.path.isdir(args.cloud):
        draw_status["draw_sequence"] = True
        cloud_list = sorted(os.listdir(args.cloud))
        for i in range(len(cloud_list)):
            if not (cloud_list[i].endswith('.bin') or \
                    cloud_list[i].endswith('.pcd') or \
                    cloud_list[i].endswith('.txt') or \
                    cloud_list[i].endswith('.npy')):
                warnings.warn('The Cloud: ', cloud_list[i], ' is not bin, pcd, txt and npy file')
                break
            if i == len(cloud_list) - 1:
                # Reach the end of cloud list
                if cloud_list[i].endswith('.bin'):
                    cloud_format["bin_cloud"] = True
                elif cloud_list[i].endswith('pcd'):
                    cloud_format["pcd_cloud"] = True
                elif cloud_list[i].endswith('txt'):\
                    cloud_format["txt_cloud"] = True
                draw_status["draw_cloud"] = True
    elif os.path.isfile(args.cloud):
        draw_status["draw_frame"] = True
        if args.cloud.endswith('.bin'):
            draw_status["bin_cloud"] = True
            draw_status["draw_cloud"] = True
            cloud_list.append(args.cloud)
        elif args.cloud.endswith('pcd'):
            draw_status["pcd_cloud"] = True
            draw_status["draw_cloud"] = True
            cloud_list.append(args.cloud)
        elif args.cloud.endswith('.txt'):
            draw_status["txt_cloud"] = True
            draw_status["draw_cloud"] = True
            cloud_list.append(args.cloud)
        elif args.cloud.endswith('.npy'):
            draw_status["npy_cloud"] = True
            draw_status["draw_cloud"] = True
            cloud_list.append(args.cloud)
        else:
            warnings.warn('The Cloud is neither pcd file nor bin file')

    # Step 2: Check the result
    if args.result is not None:
        if draw_status["draw_sequence"] and os.path.isdir(args.result):
            result_list = sorted(os.listdir(args.result))
            if len(result_list) == len(cloud_list):
                draw_status["draw_result"] = True
            else:
                warnings.warn('The results are not matching the clouds')
        elif draw_status["draw_frame"] and os.path.isfile(args.result):
            draw_status["draw_result"] = True
        else:
            warnings.warn('The results are not matching the clouds')

    # Step 3: Check the label
    if args.label is not None:
        if draw_status["draw_sequence"] and os.path.isdir(args.label):
            label_list = sorted(os.listdir(args.label))
            if len(label_list) == len(cloud_list):
                draw_status["draw_label"] = True
            else:
                warnings.warn('The labels are not matching the clouds')
        elif draw_status["draw_frame"] and os.path.isfile(args.label):
            draw_status["draw_label"] = True
        else:
            warnings.warn('The labels are not matching the clouds')

    # Step 4: Check the poly
    if args.poly is not None:
        if draw_status["draw_sequence"] and os.path.isdir(args.poly):
            poly_list = sorted(os.listdir(args.poly))
            if len(poly_list) == len(cloud_list):
                draw_status["draw_poly"] = True
            else:
                warnings.warn('The polygon results are not matching the clouds')
        elif draw_status["draw_frame"] and os.path.isfile(args.poly):
            draw_status["draw_poly"] = True
        else:
            warnings.warn('The polygon results are not matching the clouds')

    # Step 5: Check the voxel
    if args.voxel is not None:
        if draw_status["draw_sequence"] and os.path.isdir(args.voxel):
            voxel_list = sorted(os.listdir(args.voxel))
            if len(voxel_list) == len(cloud_list):
                draw_status["draw_voxel"] = True
            else:
                warnings.warn('The voxel maps are not matching the clouds')
        elif draw_status["draw_frame"] and os.path.isfile(args.voxel):
            draw_status["draw_voxel"] = True
        else:
            warnings.warn('The voxel maps are not matching the clouds')

    # Step 6: Check the image
    if args.image is not None:
        if draw_status["draw_sequence"] and os.path.isdir(args.image):
            image_list = sorted(os.listdir(args.image))
            if len(image_list) == len(cloud_list):
                draw_status["draw_image"] = True
            else:
                warnings.warn('The images are not matching the clouds')
        elif draw_status["draw_frame"] and os.path.isfile(args.image):
            draw_status["draw_image"] = True
        else:
            warnings.warn('The images are not matching the clouds')

    # Step 7: Check other flags
    if args.colorize_by_intensity:
        draw_status["draw_intensity"] = True
    if args.draw_scale:
        draw_status["draw_scale"] = True
    if args.draw_ground:
        draw_status["draw_ground"] = True
    # if args.use_screenshot:
    #     draw_status["use_screenshot"] = True
    if args.debug:
        draw_status["debug"] = True

    return draw_status, cloud_format

def set_views(args):
    voxel_size = (0.12, 0.12, 0.2)
    # area_scope = [[-72, 92], [-72, 72], [-5, 5]]
    # det_scope = [[-60, 60], [-60, 60]]

    area_scope = [[-60, 220], [-72, 72], [-5, 5]]
    det_scope = [[-60, 220], [-72, 72]]
    # if args.viewpoint.lower() == "vehicle":
    #     area_scope = [[-72, 92], [-72, 72], [-5, 5]]
    #     det_scope = [[-40, 80], [-60, 60]]
    if args.viewpoint.lower() == "v2x":
        area_scope = [[-20, 80], [-80, 80], [-5, 5]]
        det_scope = [[-60, 60], [-60, 60]]
    return area_scope, det_scope, voxel_size

def get_draw_list(args, draw_status):
    # Load the results and labels
    draw_lists = {
        "cloud_list" : [],
        "voxel_list" : [],
        "image_list" : [],
    }
    draw_elements = {
        "results"   : [],
        "labels"    : [],
        "polys"     : [],
        "voxels"    : [],
    }

    if draw_status["draw_frame"]:
        if draw_status["draw_cloud"]:
            draw_lists["cloud_list"].append(args.cloud)
        if draw_status["draw_voxel"]:
            draw_lists["voxel_list"].append(args.voxel)
        if draw_status["draw_image"]:
            draw_lists["image_list"].append(args.image)
        if draw_status["draw_result"]:
            draw_elements["results"] = data_loader.load_single_result(args.result)
        if draw_status["draw_label"]:
            draw_elements["labels"] = data_loader.load_single_label(args.label)
        if draw_status["draw_poly"]:
            draw_elements["polys"] = data_loader.oad_single_poly(args.poly)

    elif draw_status["draw_sequence"]:
        if draw_status["draw_cloud"]:
            draw_lists["cloud_list"] = data_loader.get_cloud_list(args.cloud, args.sort_by_num)
        if draw_status["draw_voxel"]:
            draw_lists["voxel_list"] = data_loader.get_voxel_list(args.voxel, args.sort_by_num)
        if draw_status["draw_image"]:
            draw_lists["image_list"] = data_loader.get_image_list(args.image, args.sort_by_num)
        if draw_status["draw_result"]:
            draw_elements["results"] = data_loader.load_results(args.result, args.sort_by_num)
        if draw_status["draw_label"]:
            draw_elements["labels"] = data_loader.load_labels(args.label, args.sort_by_num)
        if draw_status["draw_poly"]:
            draw_elements["polys"] = data_loader.load_polys(args.poly, args.sort_by_num)
    return draw_lists, draw_elements


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cloud Visualizer')
    parser.add_argument('-c', '--cloud', type=str, default=None, help='Cloud path (file or directory)')
    parser.add_argument('-i', '--image', type=str, default=None, help='Image path (file or directory)')
    parser.add_argument('-l', '--label', type=str, default=None, help='Label path (file or directory)')
    parser.add_argument('-r', '--result', type=str, default=None, help='Result path (file or directory)')
    parser.add_argument('-p', '--poly', type=str, default=None, help='Polygon result path (file or directory)')
    parser.add_argument('-v', '--voxel', type=str, default=None, help='Voxel map path (file or directory)')

    parser.add_argument('--visual', type=str, default='visual_image', help='Save path for BEV or 3D maps (file or directory)')
    parser.add_argument('--viewpoint',type=str, default='Vehicle', help='View points (Vehicle, V2X)')
    parser.add_argument('--draw_3d', action='store_true', default=False, help='Draw 3D maps')
    parser.add_argument('--draw_scale', action='store_true', default=False, help='Draw the circle scales on BEV map')
    parser.add_argument('--draw_ground', action='store_true', default=False, help='Draw ground plane on 3D map')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--ground_height', type=float, default=-1.60, help='Ground height')
    parser.add_argument('--sort_by_num', action='store_true', default=False, help='Sort the files by number instead of name')
    parser.add_argument('--video_path', type=str, default=None, help='Video path')
    parser.add_argument('--colorize_by_intensity', action='store_true', default=False, help='Colorize the BEV cloud by intensity')

    args = parser.parse_args()

    # Settings
    cls_dict, cls_array, colormap = get_cls()
    area_scope, det_scope, voxel_size = set_views(args)
    draw_status, cloud_format = check_args(args)
    draw_lists, draw_elements = get_draw_list(args, draw_status)

    if not args.draw_3d:
        # Draw BEV maps
        bev_visualizer = VisualBEV(voxel_size=voxel_size, area_scope=area_scope, colormap=colormap)
        bev_visualizer.visualization(draw_status, draw_lists, draw_elements, args.visual)

    else:
        # Draw 3D maps
        scene_visualizer = Visual3D(voxel_size=voxel_size, area_scope=area_scope, 
            colormap=colormap, viewpoint = args.viewpoint)
        scene_visualizer.visualization(draw_status, draw_lists, draw_elements, args.visual)
        
    # # Write the images to video
    # if draw_sequence and args.video_path is not None:
    #     # Create the video directory
    #     if not os.path.exists(os.path.dirname(args.video_path)):
    #         os.makedirs(os.path.dirname(args.video_path))
    #     if not args.draw_3d:
    #         image_list = os.listdir(args.visual)
    #     else:
    #         image_list = os.listdir(args.visual)
    #     if args.sort_by_num:
    #         image_list = sorted(image_list, key=lambda x: int(x.split('.')[0]))
    #     else:
    #         image_list = sorted(image_list)

    #     # Initialize the video object
    #     cloud_image = cv2.imread(os.path.join(args.visual, image_list[0]))
    #     frame_height, frame_width, frame_channels = cloud_image.shape

    #     # Check the video path
    #     video_dir = os.path.dirname(args.video_path)
    #     if not os.path.exists(video_dir):
    #         os.makedirs(video_dir)
    #     if not args.video_path.endswith('.avi'):
    #         args.video_path = args.video_path.split('.')[0] + '.avi'
    #     cloud_video = cv2.VideoWriter(args.video_path, cv2.VideoWriter_fourcc(*'MPEG'), 10,
    #                                   (frame_width,frame_height))
    #     for index, image_fn in enumerate(image_list):
    #         if index < args.beg_index or index >= args.end_index:
    #             continue
    #         cloud_video.write(cv2.imread(os.path.join(args.visual, image_fn)))
    #     cv2.destroyAllWindows()
    #     cloud_video.release()
