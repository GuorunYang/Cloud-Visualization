import json
import numpy as np
import pickle as pkl
import os
import argparse
from tqdm import tqdm

class LabelParser(object):
    """docstring for ClassName"""
    def __init__(self):
        # super(ClassName, self).__init__()
        # self.arg = arg
        self.cls_dict = {
            "smallmot"      : "Vehicle",
            "bigmot"        : "Vehicle",
            "trafficcone"   : "Misc",
            "pedestrian"    : "Pedestrian",
            "crashbarrel"   : "Misc",
            "tricyclist"    : "Cyclist",
            "bicyclist"     : "Cyclist",
            "motorcyclist"  : "Cyclist",
            "onlybicycle"   : "Cyclist",
            "crowd"         : "Misc",
            "onlytricycle"  : "Cyclist",
            "stopbar"       : "Unknown",
            "smallmovable"  : "Misc",
            "safetybarrier" : "Unknown",
            "smallunmovable": "Misc",
            "warningpost"   : "Misc",
            "fog"           : "Unknown",
            "sign"          : "Misc",
        }
        self.annotation_cls = list(self.cls_dict.keys())
        self.model_cls = [
            "Vehicle", "Pedestrian", "Cyclist", "Misc", "Unknown"
        ]


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
                anno_cls = self.cls_dict[anno["original_lbl"].lower()]
                anno_vec = np.array([anno["x"], anno["y"], anno["z"]])
                anno_dist = np.linalg.norm(anno_vec)
                if anno_dist > 400:
                    print(anno_vec)
                    continue
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
            label_cls = get_cls_mapping(anno["original_lbl"])
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
    parser.add_argument('--pkl_path', type=str, default='visual_image', help='Save path for BEV or 3D maps (file or directory)')
    args = parser.parse_args()

    label_parser = LabelParser()
    frame_num, pkl_stat = label_parser.parse_pkl_label(args.pkl_path)
    print("Frame number: ", frame_num)
    for stat_k, stat_v in pkl_stat.items():
        print("Cls: {} Count: {} Max Dist: {:.2f}".format(
            stat_k, stat_v["count"], stat_v["max_dist"]
        ))

