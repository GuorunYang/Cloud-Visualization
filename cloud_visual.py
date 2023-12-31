import os
import argparse
import warnings
import cv2
import numpy as np
import pickle5 as pkl
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
        "draw_gt_check" : False,
        "draw_poly"     : False,
        "draw_voxel"    : False,
        "draw_image"    : False,
        "draw_scale"    : False, 
        "draw_intensity": False,
        "draw_ground"   : False,
        "debug" : False,
    }
    cloud_format = {
        "bin_cloud" : False,
        "pcd_cloud" : False,
        "txt_cloud" : False,
        "npy_cloud" : False,
    }
    cloud_list = []
    # Step 1: Check the cloud and load clou list
    if args.cloud is not None:
        if data_loader.check_file_path(args.cloud):
            print("Cloud {} exist".format(args.cloud))
            if args.cloud.startswith("tos://"):
                local_cloud_path = os.path.join(args.save, "cloud")
                print("Downloading cloud from {} to {} ...".format(args.cloud, local_cloud_path))
                data_loader.download_remote_file(args.cloud, local_cloud_path)
                print("Download Finish")
                args.cloud = local_cloud_path
        else:
            print("Cannot find cloud from {}".format(args.cloud))
    else:
        print("Argument cloud is not provided!!!")

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
                elif cloud_list[i].endswith('txt'):
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
        if data_loader.check_file_path(args.result):
            draw_status["draw_result"] = True
        else:
            draw_status["draw_result"] = False

    # Step 3: Check the label
    if args.label is not None:
        local_label_path = None
        if data_loader.check_file_path(args.label):
            if args.label.startswith("tos://"):
                label_fn = os.path.basename(args.label)
                local_label_dir = os.path.join(args.save, "label")
                print("Downloading label from {} to {} ...".format(args.label, local_label_dir))
                data_loader.download_remote_file(args.label, local_label_dir)
                local_label_path = os.path.join(local_label_dir, label_fn)
                print("Download Finish")
                args.label = local_label_path
            draw_status["draw_label"] = True
        else:
            draw_status["draw_label"] = False

    # Step 4: check the gt
    if args.gt_check is not None:
        if data_loader.check_file_path(args.gt_check):
            check_fn = os.path.basename(args.gt_check)
            local_check_dir = os.path.join(args.save, "gt_check")
            print("Downloading gt check info from {} to {} ...".format(args.gt_check, local_check_dir))
            data_loader.download_remote_file(args.gt_check, local_check_dir)
            local_check_path = os.path.join(local_check_dir, check_fn)
            print("Download Finish")
            args.gt_check = local_check_path
            draw_status["draw_gt_check"] = True
        else:
            draw_status["draw_gt_check"] = False

    # Step 5: Check the poly
    if args.poly is not None:
        if data_loader.check_file_path(args.poly):
            draw_status["draw_poly"] = True
        else:
            draw_status["draw_poly"] = False

    # Step 6: Check the voxel
    if args.voxel is not None:
        if data_loader.check_file_path(args.voxel):
            draw_status["draw_voxel"] = True
        else:
            draw_status["draw_voxel"] = False

    # Step 7: Check the image
    if args.image is not None:
        if data_loader.check_file_path(args.image):
            draw_status["draw_image"] = True
        else:
            draw_status["draw_image"] = False

    # Step 8: Check other flags
    if args.colorize_by_intensity:
        draw_status["draw_intensity"] = True
    if args.draw_scale:
        draw_status["draw_scale"] = True
    if args.draw_ground:
        draw_status["draw_ground"] = True
    if args.debug:
        draw_status["debug"] = True
    return draw_status, cloud_format

def set_views(args):
    voxel_size = (0.12, 0.12, 0.2)
    # area_scope = [[-72, 92], [-72, 72], [-5, 5]]
    # det_scope = [[-60, 60], [-60, 60]]

    area_scope = [[-80, 220], [-72, 72], [-5, 5]]
    det_scope = [[-60, 220], [-25.6, 25.6]]
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
        "checks"    : [],
    }

    if draw_status["draw_frame"]:
        if draw_status["draw_cloud"]:
            draw_lists["cloud_list"].append(args.cloud)
        if draw_status["draw_voxel"]:
            draw_lists["voxel_list"].append(args.voxel)
        if draw_status["draw_image"]:
            draw_lists["image_list"].append(args.image)
        query_frame = os.path.splitext(args.cloud.split("/")[-1])[0]
        if draw_status["draw_result"]:
            draw_elements["results"] = data_loader.load_results(args.result, query_frame=query_frame)
        if draw_status["draw_label"]:
            draw_elements["labels"] = data_loader.load_labels(args.label, query_frame=query_frame)
        if draw_status["draw_gt_check"]:
            draw_elements["checks"] = data_loader.load_checks(args.gt_check, query_frame=query_frame)
        if draw_status["draw_poly"]:
            draw_elements["polys"] = data_loader.load_single_poly(args.poly)


    elif draw_status["draw_sequence"]:
        if draw_status["draw_cloud"]:
            draw_lists["cloud_list"] = data_loader.get_cloud_list(args.cloud, args.sort_by_num)
        if draw_status["draw_voxel"]:
            draw_lists["voxel_list"] = data_loader.get_voxel_list(args.voxel, args.sort_by_num)
        if draw_status["draw_image"]:
            draw_lists["image_list"] = data_loader.get_image_list(args.image, args.sort_by_num)
        if draw_status["draw_result"]:
            draw_elements["results"] = data_loader.load_results(args.result, None, args.sort_by_num)
        if draw_status["draw_label"]:
            draw_elements["labels"] = data_loader.load_labels(args.label, None, args.sort_by_num)
        if draw_status["draw_poly"]:
            draw_elements["polys"] = data_loader.load_polys(args.poly, args.sort_by_num)
        if draw_status["draw_gt_check"]:
            draw_elements["checks"] = data_loader.load_checks(args.gt_check, None, args.sort_by_num)
    return draw_lists, draw_elements


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cloud Visualizer')
    parser.add_argument('-c', '--cloud', type=str, default=None, help='Cloud path (file or directory)')
    parser.add_argument('-i', '--image', type=str, default=None, help='Image path (file or directory)')
    parser.add_argument('-l', '--label', type=str, default=None, help='Label path (file or directory)')
    parser.add_argument('-r', '--result', type=str, default=None, help='Result path (file or directory)')
    parser.add_argument('-p', '--poly', type=str, default=None, help='Polygon result path (file or directory)')
    parser.add_argument('-v', '--voxel', type=str, default=None, help='Voxel map path (file or directory)')
    parser.add_argument('-g', '--gt_check', type=str, default=None, help='GT check path (file)')

    parser.add_argument('--save', type=str, default='./', help='Save path for remote data')    
    parser.add_argument('--visual', type=str, default='visual_image', help='Save path for BEV or 3D maps (file or directory)')
    parser.add_argument('--viewpoint', type=str, default='Vehicle', help='View points (Vehicle, V2X)')
    parser.add_argument('--draw_3d', action='store_true', default=False, help='Draw 3D maps')
    parser.add_argument('--draw_scale', action='store_true', default=False, help='Draw the circle scales on BEV map')
    parser.add_argument('--draw_ground', action='store_true', default=False, help='Draw ground plane on 3D map')
    parser.add_argument('--rotation', type=float, default=0.0, help='Rotation angle')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--ground_height', type=float, default=-1.60, help='Ground height')
    parser.add_argument('--sort_by_num', action='store_true', default=False, help='Sort the files by number instead of name')
    parser.add_argument('--colorize_by_intensity', action='store_true', default=False, help='Colorize the BEV cloud by intensity')

    args = parser.parse_args()

    # Settings
    area_scope, det_scope, voxel_size = set_views(args)
    draw_status, cloud_format = check_args(args)
    # print("Draw Status: ", draw_status)
    draw_lists, draw_elements = get_draw_list(args, draw_status)

    if not args.draw_3d:
        # Draw BEV maps
        bev_visualizer = VisualBEV(voxel_size=voxel_size, area_scope=area_scope, 
                                   colormap=data_loader.colormap, cloud_rotation=args.rotation)
        bev_visualizer.visualization(draw_status, draw_lists, draw_elements, args.visual)

    else:
        # Draw 3D maps
        scene_visualizer = Visual3D(voxel_size=voxel_size, area_scope=area_scope, 
                                    colormap=data_loader.colormap, viewpoint = args.viewpoint, cloud_rotation=args.rotation)
        scene_visualizer.visualization(draw_status, draw_lists, draw_elements, args.visual)
