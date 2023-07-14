import json
import numpy as np
import pickle as pkl
import os
import cv2
import argparse
import data_loader
import matplotlib.pyplot as plt
from tqdm import tqdm

class LabelParser(object):
    """docstring for ClassName"""
    def __init__(self):
        # super(ClassName, self).__init__()
        # self.arg = arg
        # self.cls_dict = {
        #     "smallmot"      : "Vehicle",
        #     "bigmot"        : "Vehicle",
        #     "trafficcone"   : "Misc",
        #     "pedestrian"    : "Pedestrian",
        #     "crashbarrel"   : "Misc",
        #     "tricyclist"    : "Cyclist",
        #     "bicyclist"     : "Cyclist",
        #     "motorcyclist"  : "Cyclist",
        #     "onlybicycle"   : "Cyclist",
        #     "crowd"         : "Misc",
        #     "onlytricycle"  : "Cyclist",
        #     "stopbar"       : "Unknown",
        #     "smallmovable"  : "Misc",
        #     "safetybarrier" : "Unknown",
        #     "smallunmovable": "Misc",
        #     "warningpost"   : "Misc",
        #     "fog"           : "Unknown",
        #     "sign"          : "Misc",
        # }
        self.cls_dict = data_loader.cls_dict
        self.annotation_cls = list(self.cls_dict.keys())
        self.model_cls = {
            "Vehicle"       : 0,
            "Pedestrian"    : 1, 
            "Cyclist"       : 2, 
            "Misc"          : 3, 
            "Unknown"       : 4,
        }
        


    def filter_pkl_label(self, 
                         src_label_path, 
                         des_label_path,
                         reserve_frame_dir = None,  
                         filter_num = -1):
        pkl_file = open(src_label_path, 'rb')
        src_label = pkl.load(pkl_file)
        des_label = {}
        frame_num = len(src_label)
        reserve_frames = set()
        det_cls_array = list(data_loader.det_cls_dict.values())
        print("Det cls array: ", det_cls_array)
        if reserve_frame_dir is not None:
            if os.path.isdir(reserve_frame_dir):
                reserve_list = os.listdir(reserve_frame_dir)
                for reserve_name in reserve_list:
                    reserve_frames.add(reserve_name.split(".")[0])
        for frame_name, frame_label in tqdm(src_label.items()):
            if filter_num > 0:
                anno_cnt = 0
                for k, anno in enumerate(frame_label):
                    try:
                        anno_cls = self.cls_dict[anno["original_lbl"].lower()]
                    except:
                        print("Origin lbl: {}, Mapping lbl: {}".format(
                            anno["original_lbl"], anno["lbl"]
                        ))
                    if (anno_cls in det_cls_array) and (anno["is_crowd"] == 'False'):
                        anno_cnt += 1
                # print("Annos count:", anno_cnt)
                if anno_cnt <= filter_num:
                    print("{}".format(frame_name))
                    continue
            if len(reserve_frames) > 0:
                if frame_name not in reserve_frames:
                    continue
            des_label[frame_name] = frame_label
        print("Filter frame: {} -> {}".format(len(src_label), len(des_label)))
        with open(des_label_path, "wb") as f:
            pkl.dump(des_label, f)


    def parse_pkl_label(self, pkl_label_path, kitti_label_dir = None):
        '''
            Parse the pkl label as KITTI format
        '''
        label_dict = {}
        # Check the label path and directory
        if not os.path.exists(pkl_label_path):
            print('PKL label path: ', pkl_label_path, ' does not exist!')
            return label_dict

        cls_stat = {}
        for cls_name in self.model_cls:
            cls_stat[cls_name] = {
                "count" : 0, 
                "max_dist" : 0
            }

        # Load the pkl labels
        pkl_file = open(pkl_label_path, 'rb')
        pkl_label = pkl.load(pkl_file)
        frame_num = len(pkl_label)
        for frame_name, frame_label in tqdm(pkl_label.items()):
            for k, anno in enumerate(frame_label):
                # print("ori lbl: ", anno["original_lbl"].lower())
                try:
                    anno_cls = self.cls_dict[anno["original_lbl"].lower()]
                except:
                    print("Origin lbl: {}, Mapping lbl: {}".format(
                        anno["original_lbl"], anno["lbl"]
                    ))
                anno_vec = np.array([anno["x"], anno["y"], anno["z"]])
                anno_dist = np.linalg.norm(anno_vec)
                # if anno_dist > 400:
                #     print(anno_vec)
                #     continue
                if anno_cls in cls_stat:
                    # print("Cls stat: ", cls_stat[anno_cls])
                    cls_stat[anno_cls]["count"] += 1
                    if anno_dist > cls_stat[anno_cls]["max_dist"]:
                        cls_stat[anno_cls]["max_dist"] = anno_dist        
        # with open(pkl_label_path, 'rb') as f:
        #     pkl_label = pkl.load(f)
        #     # print("Label: ", pkl_label)
        #     if not kitti_label_dir is None:
        #         os.makedirs(kitti_label_dir, exist_ok=True)

        #     for frame_name, frame_label in tqdm(pkl_label.items()):
        #         for k, anno in enumerate(frame_label):
        #             anno_cls = self.cls_dict(anno["original_lbl"].lower())
        #             anno_vec = np.array([anno["x"], anno["y"], anno["z"]])
        #             anno_dist = np.linalg.norm(anno_vec)
        #             if anno_cls in cls_stat:
        #                 cls_stat[anno_cls]["count"] += 1
        #                 if anno_dist > cls_stat[anno_cls]["max_dist"]:
        #                     cls_stat[anno_cls]["max_dist"] = anno_dist
        return frame_num, cls_stat


    def draw_label_map(self, pkl_label_path, save_dir, 
                       draw_range = [-100, 100, -200, 400], # [ymin, ymax, xmin, xmax]
                       voxel_size = 5, 
                       step_meter = 50):
        # Initialize the label count
        size_x = int((draw_range[3] - draw_range[2]) / voxel_size)
        size_y = int((draw_range[1] - draw_range[0]) / voxel_size)
        # label_count = np.zeros((5, size_y, size_x), dtype = int)
        label_count = np.zeros((5, size_x, size_y), dtype = int)
        print("Label count size: ", label_count.shape)
        os.makedirs(save_dir, exist_ok=True)

        label_dict = {}
        # Check the label path and directory
        if not os.path.exists(pkl_label_path):
            print('PKL label path: ', pkl_label_path, ' does not exist!')
            return label_dict

        # Load the pkl labels
        pkl_file = open(pkl_label_path, 'rb')
        pkl_label = pkl.load(pkl_file)
        frame_num = len(pkl_label)
        for frame_name, frame_label in tqdm(pkl_label.items()):
            for k, anno in enumerate(frame_label):
                # print("ori lbl: ", anno["original_lbl"].lower())
                # annos_cls = None
                anno_cls = self.cls_dict[anno["original_lbl"].lower()]
                anno_cls_id = self.model_cls[anno_cls]
                x, y = anno['x'], anno['y']
                x_ind = int((x - draw_range[2]) / voxel_size)
                y_ind = int((y - draw_range[0]) / voxel_size)
                # print("Cls: {} x: {}, y: {}".format(anno_cls, x_ind, y_ind))
                # print("Size: {} {}".format(size_x, size_y))
                if (x_ind < size_x) and (y_ind < size_y):
                    label_count[anno_cls_id, x_ind, y_ind] += 1
                    label_count[-1, x_ind, y_ind] += 1
        label_count = label_count.astype(np.float32)
        for i in range(label_count.shape[0]):
            max_count = np.max(label_count[i, :])
            # print("Max count: ", max_count)
            label_count[i, :] = label_count[i, :] / max_count
        
        cmap = plt.get_cmap('viridis')
        for cls_name, cls_id in self.model_cls.items():
            save_pth = os.path.join(save_dir, cls_name + '.png')
            label_count[cls_id, :] = np.flipud(label_count[cls_id, :])
            colorize_label_count = cmap(label_count[cls_id, :]) * 255.0
            colorize_label_count = colorize_label_count.astype(int)
            # colorize_label_count = cv2.cvtColor(colorize_label_count, cv2.COLOR_BGR2RGB)
            # plt.figure(dpi=80)
            fig, ax = plt.subplots()
            ax_step = step_meter / voxel_size
            ax.imshow(colorize_label_count)
            ax.set_xlim([0, colorize_label_count.shape[1]])
            ax.set_ylim([colorize_label_count.shape[0], 0])
            # print("Xticks:   ", np.arange(0, colorize_label_count.shape[1]+ax_step, ax_step/voxel_size))
            # print("Xlabels : ", np.arange(draw_range[0], draw_range[1]+ax_step, ax_step))
            ax.set_xticks(np.arange(0, colorize_label_count.shape[1]+ax_step, ax_step))
            ax.set_yticks(np.arange(0, colorize_label_count.shape[0]+ax_step, ax_step))
            ax.set_xticklabels(np.arange(draw_range[0], draw_range[1]+step_meter, step_meter))
            ax.set_yticklabels(np.arange(draw_range[3], draw_range[2]-step_meter, -step_meter))
            # ax.set_yticks(np.arange(draw_range[2], draw_range[3], 50))
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(True)
            plt.title('Cls label {} distribution'.format(cls_name))
            plt.savefig(save_pth)
            # plt.show()
            xmin, xmax, ymin, ymax = draw_range[0], draw_range[1], draw_range[2], draw_range[3]
            # plt.rcParams["figure.autolayout"] = True
            # plt.imshow(colorize_label_count, extent=[xmin, xmax, ymin, ymax])
            # plt.xlim((draw_range[0], draw_range[1]))
            # plt.ylim((draw_range[2], draw_range[3]))
            # plt.xticks(x_ticks)
            # plt.yticks(y_ticks)
            # plt.show()
            # plt.savefig(save_pth)
            # cv2.imwrite(save_pth, colorize_label_count)



    def labels2lines(self, labels):
        '''
            For pkl labels:
               'dimension': (length, height, width)
            For KITTI label line:
                [0-3]:      type truncated occluded alpha
                [4-7]:      2D bbox (left, top, right, bottom)
                [8-10]:     3D dimension (height, width, length)
                [11-13]:    3D locatoin (x, y, z)
                [14]:       3D rotation_y
                [15]:       score (optional)
        '''

        lines = []
        for i, anno in enumerate(labels):
            # label_cls = anno["original_lbl"]
            anno["rotation_y"] = np.pi / 2.0 - anno["rotation_y"] + np.pi
            label_cls = self.cls_dict[anno["original_lbl"]]
            label_dimension = "{:.4f} {:.4f} {:.4f}".format(anno["height"], anno["width"], anno["length"])
            label_location = "{:.4f} {:.4f} {:.4f}".format(anno["x"], anno["y"], anno["z"])
            label_rotation = "{:.4f}".format(anno["rotation_y"])
            label_2D = "0.00 0.00 0.00 0.00"
            label_occluded = "0.00"
            label_truncated = "0.00"
            label_alpha = "{:.4f}".format(-np.arctan2(-anno["y"], anno["x"]) + anno["rotation_y"])
            label_ln = "{} {} {} {} {} {} {} {}\n".format(label_cls, label_occluded, label_truncated, label_alpha, label_2D, \
                label_dimension, label_location, label_rotation)
            lines.extend(label_ln)
        return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label Parser')
    parser.add_argument('--src_pkl_path', type=str, default="annos.pkl", help='Src pkl path')
    parser.add_argument('--des_pkl_path', type=str, default=None, help='Des pkl path')
    parser.add_argument('--reserve_dir', type=str, default=None, help='Reserve frame directory')
    parser.add_argument('--filter_num', type=int, default=-1, help='Filter num')
    parser.add_argument('--draw_label', action='store_true', default=False, help='Draw label distribution')
    parser.add_argument('--draw_dir', type=str, default="./draw_label", help='Directory of draw label')

    args = parser.parse_args()

    label_parser = LabelParser()
    frame_num, pkl_stat = label_parser.parse_pkl_label(args.src_pkl_path)
    print("Frame number: ", frame_num)
    for stat_k, stat_v in pkl_stat.items():
        print("Cls: {} Count: {} Max Dist: {:.2f}".format(
            stat_k, stat_v["count"], stat_v["max_dist"]
        ))
    if args.draw_label:
        label_parser.draw_label_map(args.src_pkl_path, args.draw_dir)
    if args.des_pkl_path is not None:  
        label_parser.filter_pkl_label(args.src_pkl_path, args.des_pkl_path, args.reserve_dir, args.filter_num)
