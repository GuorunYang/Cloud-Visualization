import os
import cv2
import warnings
import numpy as np
import matplotlib.pyplot as plt
import data_loader
from tqdm import tqdm

class VisualBEV(object):
    """
        BEV Visualization for point cloud and detections
    """
    def __init__(self, voxel_size = None, area_scope = None, colormap = None):
        self.voxel_size = (0.12, 0.12, 0.2)
        self.area_scope = [[-72, 92], [-72, 72], [-5, 5]]
        self.colormap = [
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
        if not voxel_size is None:
            self.voxel_size = voxel_size
        if not area_scope is None:
            self.area_scope = area_scope
        if not colormap is None:
            self.colormap = colormap


    def visualization(self, draw_status, draw_lists, draw_elements, save_path):
        if draw_status["draw_frame"]:
            frame_cloud_path = draw_lists["cloud_list"][0]
            frame_results, frame_labels, frame_polys = None, None, None
            frame_voxel_path, frame_image_path = None, None
            frame_name = os.path.splitext(os.path.basename(frame_cloud_path))[0]
            if draw_status["draw_result"]:
                if frame_name in draw_elements["results"]:
                    frame_results = draw_elements["results"][frame_name]
                else:
                    print("Cannot find frame {} in results".format(frame_name))
            if draw_status["draw_label"]:
                if frame_name in draw_elements["labels"]:
                    frame_labels = draw_elements["labels"][frame_name]
                else:
                    print("Cannot find frame {} in labels".format(frame_name))
            # if draw_status["draw_poly"]:
            #     frame_polys = draw_elements["polys"][frame_name]
            # if draw_status["draw_voxel"]:
            #     frame_voxel_path = draw_lists["voxel_list"][frame_name]
            # if draw_status["draw_image"]:
            #     frame_image_path = draw_lists["image_list"][frame_name]
            frame_bev_map = self.draw_bev_map(draw_status, frame_cloud_path, frame_results, frame_labels, 
                frame_polys, frame_voxel_path, frame_image_path)
            if draw_status["draw_scale"]:
                frame_bev_map = self.draw_bev_circle_scale(frame_bev_map, color = (200, 200, 200))
            frame_bev_path = (frame_cloud_path.split('/')[-1]).split('.')[0] + '.png'
            if os.path.isdir(save_path):
                # If the directory is provided, the map is writed into the directory
                frame_bev_fn = (frame_cloud_path.split('/')[-1]).rsplit('.', 1)[0] + '.png'
                frame_bev_path = os.path.join(save_path, frame_bev_fn)
                print('Visualized Frame Path: ', frame_bev_path)
            elif save_path.endswith('.png'):
                # If the png path is provided, the map is save as the path
                frame_bev_path = save_path
                print('Visualized Frame Path: ', frame_bev_path)
            if not cv2.imwrite(frame_bev_path, frame_bev_map):
                print('Write Image Error! Please check the path!')
                return False

        elif draw_status["draw_sequence"]:
            os.makedirs(save_path, exist_ok=True)
            print('Visualized Sequence Path: ', save_path)
            for i in tqdm(range(len(draw_lists["cloud_list"]))):
                frame_cloud_path = draw_lists["cloud_list"][i]
                frame_cloud_name = frame_cloud_path.split("/")[-1]
                frame_results, frame_labels, frame_polys = None, None, None
                frame_voxel_path, frame_image_path = None, None
                frame_name = os.path.splitext(os.path.basename(frame_cloud_path))[0]
                if draw_status["draw_result"]:
                    if frame_name in draw_elements["results"]:
                        frame_results = draw_elements["results"][frame_name]
                    else:
                        print("Cannot find frame {} in results".format(frame_name))
                if draw_status["draw_label"]:
                    if frame_name in draw_elements["labels"]:
                        frame_labels = draw_elements["labels"][frame_name]
                    else:
                        print("Cannot find frame {} in labels".format(frame_name))
                # if draw_status["draw_poly"]:
                #     frame_polys = draw_elements["polys"][i]
                # if draw_status["draw_voxel"]:
                #     frame_voxel_path = draw_lists["voxel_list"][i]
                # if draw_status["draw_image"]:
                #     frame_image_path = draw_lists["image_list"][i]
                frame_bev_map = self.draw_bev_map(draw_status, frame_cloud_path, frame_results, frame_labels, 
                    frame_polys, frame_voxel_path, frame_image_path)
                if draw_status["draw_scale"]:
                    frame_bev_map = self.draw_bev_circle_scale(frame_bev_map, color=(200, 200, 200))
                frame_bev_fn = (frame_cloud_path.split('/')[-1]).rsplit('.', 1)[0] + '.png'
                if not cv2.imwrite(os.path.join(save_path, frame_bev_fn), frame_bev_map):
                    print('Write Image Error! Please check the path!')
                    return False
        return True


    def draw_bev_map(self, draw_status, cloud_path, frame_results = None, frame_labels = None, frame_polys = None,
                 frame_voxel_path = None, frame_image_path = None, histogram_intensity = True):
        if draw_status["draw_cloud"]:
            cloud = data_loader.load_cloud(cloud_path)
            voxel_map = self.convert_pts_to_bev_map(cloud)
            bev_map = voxel_map.sum(axis=2)

            if draw_status["draw_intensity"]:
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

            if draw_status["draw_result"] and draw_status["draw_label"]:
                det_boxes3d, det_scores, det_cls, det_trackids = data_loader.get_boxes_from_results(frame_results)
                gt_boxes3d, gt_scores, gt_cls, gt_trackids = data_loader.get_boxes_from_labels(frame_labels)
                bev_map = self.draw_bev_boxes(bev_map=bev_map.copy(), boxes3d=gt_boxes3d, labels=gt_cls,
                                         scores=gt_scores, track_ids=gt_trackids,
                                         thickness = 2, colorize_with_label = False, color=(178, 255, 102))
                bev_map = self.draw_bev_boxes(bev_map = bev_map.copy(), boxes3d = det_boxes3d, labels = det_cls,
                                         scores = det_scores, track_ids = det_trackids,
                                         thickness = 2, colorize_with_label = False, color=(45, 151, 255))
            else:
                if draw_status["draw_result"]:
                    det_boxes3d, det_scores, det_cls, det_trackids = data_loader.get_boxes_from_results(frame_results)
                    bev_map = self.draw_bev_boxes(bev_map = bev_map.copy(), boxes3d = det_boxes3d, labels = det_cls,
                                             scores = det_scores, track_ids = det_trackids)
                if draw_status["draw_label"]:
                    gt_boxes3d, gt_scores, gt_cls, gt_trackids = data_loader.get_boxes_from_labels(frame_labels)
                    bev_map = self.draw_bev_boxes(bev_map = bev_map.copy(), boxes3d = gt_boxes3d, labels = gt_cls,
                                             scores = gt_scores, track_ids = gt_trackids)
                if draw_status["draw_poly"]:
                    poly_boxes3d, poly_vertices3d, poly_cls, poly_trackids = data_loader.get_polys(frame_polys)
                    bev_map = self.draw_bev_polys(bev_map = bev_map.copy(), polys = poly_vertices3d)
            bev_image = bev_map.astype(np.uint8)
            if draw_status["draw_voxel"]:
                bev_image = self.draw_bev_voxels(bev_image, voxel_path = frame_voxel_path)
            if draw_status["draw_image"]:
                bev_image = self.draw_image(bev_image, image_path = frame_image_path, image_ratio = 0.2)
            return bev_image


    def draw_bev_voxels(self, img, voxel_path, 
                        coor_scale = 0.16, 
                        center_voxel = [532, 320], 
                        voxel_color = (244, 0, 0)):
        '''
            Draws voxels on bev map
        '''
        # Get the voxels from voxel map
        voxel_image = cv2.imread(voxel_path, cv2.IMREAD_COLOR)
        other_index = np.where(voxel_image[:, :, 0] == 250)
        index_array = np.asarray(other_index).transpose()[:, 0:2]
        # Transform the voxels to bev pixels
        voxel_points = np.zeros([index_array.shape[0], 3], dtype = np.float)
        for i in range(index_array.shape[0]):
            voxel_row, voxel_col = index_array[i][0], index_array[i][1]
            voxel_i = (voxel_row - center_voxel[0]) * -1.0
            voxel_j = (voxel_col - center_voxel[1]) * -1.0
            point_x = voxel_i * coor_scale
            point_y = voxel_j * coor_scale
            voxel_points[i] = np.array([point_x, point_y, 0.0], dtype=np.float)
        bev_pixels = self.pt_to_bev_pix(voxel_points)
        for k in range(bev_pixels.shape[0]):
            pixel_row, pixel_col = bev_pixels[k, 0], bev_pixels[k, 1]
            cv2.circle(img, (pixel_row, pixel_col), radius = 2, color = voxel_color, thickness = -1)
        return img


    def draw_image(self, img, image_path, image_ratio, 
                    alpha = 0.8,
                    upper_left = True, 
                    upper_right = False, 
                    bottom_left = False, 
                    bottom_right = False):
        camera_image = cv2.imread(image_path)
        camera_image_height, camera_image_width = camera_image.shape[0], camera_image.shape[1]
        camera_image_height = int(camera_image_height * image_ratio)
        camera_image_width = int(camera_image_width * image_ratio)
        camera_image = cv2.resize(camera_image, (camera_image_width, camera_image_height), interpolation=cv2.INTER_LINEAR)

        image_left, image_up = 0, 0
        image_right, image_down = img.shape[1], img.shape[0]
        # Draw the image on the upper-left
        if upper_right:
            image_left = image_right - camera_image_width
            image_down = image_up + camera_image_height
        elif bottom_left:
            image_right = image_left + camera_image_width
            image_up = image_down - camera_image_height
        elif bottom_right:
            image_left = image_right - camera_image_width
            image_up = image_down - camera_image_height
        elif upper_left:
            image_right = image_left + camera_image_width
            image_down = image_up + camera_image_height
        img_roi = img[image_up:image_down, image_left:image_right, :]
        bless_roi = cv2.addWeighted(img_roi, 1-alpha, camera_image, alpha, 0.0)
        img[image_up:image_down, image_left:image_right, :] = bless_roi
        return img


    def draw_icons2d(self, vertices2d, img, icon_path, icon_size, color=(0, 0, 255), max_num=500, add_height = 15):
        num = min(max_num, len(vertices2d))
        icon_img = cv2.imread(icon_path)
        icon_img = cv2.resize(icon_img, (icon_size, icon_size), interpolation=cv2.INTER_NEAREST)
        gray_img = cv2.cvtColor(icon_img, cv2.COLOR_BGR2GRAY)
        icon_mask = (gray_img != 0)
        icon_img[:, :, :] = color[:]
        icon_img[icon_mask, :] = [0, 0, 0]

        for n in range(num):
            b = vertices2d[n]  # (1,2)
            b = np.asarray(b, dtype=np.int)

            # Check the icon
            icon_left   = b[0] - int(icon_size / 2.0)
            icon_right  = b[0] + int(icon_size / 2.0)
            icon_up     = b[1]
            icon_down   = b[1] + int(icon_size)
            # if 'ped' in icon_path:
            #   icon_up = icon_up - add_height
            #   icon_down = icon_down - add_height

            if (icon_left >= 0) and (icon_left <img.shape[0]) and (icon_up >= 0) and (icon_down < img.shape[1]):
                # img_patch = img[icon_up:icon_down, icon_left:icon_right, :]
                # img_mask = (img_patch == color)
                # icon_img[img_mask, :] = color
                img[icon_up:icon_down, icon_left:icon_right, :] = icon_img[:, :, :]

            # for i in range(b[1] - icon_size, b[1] + icon_size):
            #   for j in range(b[0] - icon_size, b[0] + icon_size):
            #       if i < img.shape[0] and j < img.shape[1]:
            #           img[i][j] = color

        return img

    def draw_boxes(self, image, boxes, color=(255, 0, 0), scores=None):
        """
        :param image:
        :param boxes: (N, 4) [x1, y1, x2, y2] / (N, 8) for top boxes/ (N, 8, 2) 3D corners in image coordinate
        :return:
        """
        import cv2
        line_map = [(0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7)]  # line maps for linking 3D corners

        font = cv2.FONT_HERSHEY_SIMPLEX
        image = image.copy()
        for k in range(boxes.shape[0]):
            if boxes.shape[1] == 4:
                x1, y1, x2, y2 = boxes[k, 0], boxes[k, 1], boxes[k, 2], boxes[k, 3]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, '%.2f' % scores[k], (x1, y1), font, 1, color, 2, cv2.LINE_AA)
            elif boxes.shape[1] == 8 and boxes.shape.__len__() == 2:
                # draw top boxes
                for j in range(0, 8, 2):
                    x1, y1 = boxes[k, j], boxes[k, j + 1]
                    x2, y2 = boxes[k, (j + 2) % 8], boxes[k, (j + 3) % 8]
                    cv2.line(image, (x1, y1), (x2, y2), color, 2)
                if scores is not None:
                    cv2.putText(image, '%.2f' % scores[k], (boxes[k, 0], boxes[k, 1]), font,
                                1, color, 2, cv2.LINE_AA)
            else:
                # draw 3D corners on RGB image
                assert boxes.shape[1:] == (8, 2)
                for line in line_map:
                    x1, y1 = boxes[k, line[0], 0], boxes[k, line[0], 1]
                    x2, y2 = boxes[k, line[1], 0], boxes[k, line[1], 1]
                    cv2.line(image, (x1, y1), (x2, y2), color, 2)
                # draw on orientation face
                cv2.line(image, tuple(boxes[k, 0, :]), tuple(boxes[k, 5, :]), (0, 250, 0), 2)
                cv2.line(image, tuple(boxes[k, 1, :]), tuple(boxes[k, 4, :]), (0, 250, 0), 2)

                pt_idx = 4
                x1, y1 = boxes[k, pt_idx, 0], boxes[k, pt_idx, 1]
                if scores is not None:
                    cv2.putText(image, '%s' % scores[k], (x1, y1), font, 1, color, 2, cv2.LINE_AA)

        return image

    def boxes3d_to_corners3d_lidar(self, boxes3d):
        """
        :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords
        :param rotate:
        :return: corners3d: (N, 8, 3)
        """
        boxes_num = boxes3d.shape[0]
        w, l, h = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
        x_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
        y_corners = np.array([-l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2.], dtype=np.float32).T
        z_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        z_corners[:, 4:8] = h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)

        ry = boxes3d[:, 6]
        zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
        rot_list = np.array([[np.cos(ry), -np.sin(ry), zeros],
                             [np.sin(ry), np.cos(ry), zeros],
                             [zeros, zeros, ones]])  # (3, 3, N)
        R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

        temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                       z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
        rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
        x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

        x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

        x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
        y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
        z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

        corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

        return corners.astype(np.float32)

    def create_scope_filter(self, pts):
        """
        :param pts: (N, 3) point cloud in rect camera coords
        :area_scope: (3, 2), area to keep [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
        """
        pts = pts.transpose()
        x_scope, y_scope, z_scope = self.area_scope[0], self.area_scope[1], self.area_scope[2]
        scope_filter = (pts[0] > x_scope[0]) & (pts[0] < x_scope[1]) \
                       & (pts[1] > y_scope[0]) & (pts[1] < y_scope[1]) \
                       & (pts[2] > z_scope[0]) & (pts[2] < z_scope[1])

        return scope_filter

    def convert_pts_to_bev_map(self, points):
        """
        :param pts: (N, 3 or 4) point cloud in rect camera coords
        """
        voxel_size = np.array(self.voxel_size)
        area_scope = np.array(self.area_scope)
        scope_filter = self.create_scope_filter(points)
        pts_val = points[scope_filter]
        voxel_idxs = np.floor(pts_val[:, 0:3] / voxel_size).astype(np.int32)

        min_voxel_coords = np.floor(area_scope[:, 0] / voxel_size).astype(np.int32)
        max_voxel_coords = np.ceil(area_scope[:, 1] / voxel_size - 1).astype(np.int32)

        # Check the points are bounded by the image scope
        assert (min_voxel_coords <= np.amin(voxel_idxs, axis=0)).all(), 'Shape: %s' % (str(voxel_idxs.shape))
        assert (max_voxel_coords >= np.amax(voxel_idxs, axis=0)).all(), 'Shape: %s' % (str(voxel_idxs.shape))
        voxel_idxs = voxel_idxs - min_voxel_coords

        voxelized_shape = ((max_voxel_coords - min_voxel_coords) + 1).astype(np.int32)
        if points.shape[1] == 4:
            voxelized_shape[2] += 1  # intensity channel

        L, W, H = voxelized_shape[2], voxelized_shape[0], voxelized_shape[1]
        voxel_map = np.zeros(voxelized_shape, dtype=np.float32)
        voxel_map[voxel_idxs[:, 0], voxel_idxs[:, 1], voxel_idxs[:, 2]] = 1.0

        if points.shape[1] >= 4:
            intensity = points[:, 3]
            if np.amax(intensity) <= 1.0:
                intensity *= 255.0
            intensity = intensity[scope_filter]
            for i in range(intensity.shape[0]):
                # Save biggest intensity in each voxel
                if intensity[i] > voxel_map[voxel_idxs[i, 0], voxel_idxs[i, 1], -1]:
                    # print('update the intensity from ', voxel_map[voxel_idxs[i, 0], voxel_idxs[i, 1], -1], ' to ', intensity[i])
                    voxel_map[voxel_idxs[i, 0], voxel_idxs[i, 1], -1] = intensity[i]
            # voxel_map[voxel_idxs[:, 0], voxel_idxs[:, 1], -1] = intensity  # TODO: which point to use?

        voxel_map = np.flip(np.flip(voxel_map, axis=0), axis=1)
        return voxel_map

    def corners3d_to_bev_corners(self, corners3d):
        """
        :param corners3d: (N, 8, 3)
        :return:
            bev_corners: (N, 4, 2)
        """
        voxel_size = np.array(self.voxel_size)
        area_scope = np.array(self.area_scope)

        voxel_idxs = np.floor(corners3d[:, :, 0:3] / voxel_size).astype(np.int32)

        min_voxel_coords = np.floor(area_scope[:, 0] / voxel_size).astype(np.int32)
        max_voxel_coords = np.ceil(area_scope[:, 1] / voxel_size - 1).astype(np.int32)
        voxelized_shape = ((max_voxel_coords - min_voxel_coords) + 1).astype(np.int32)

        # Check the points are bounded by the image scope
        # assert (min_voxel_coords <= np.amin(voxel_idxs, axis=0)).all(), 'Shape: %s' % (str(voxel_idxs.shape))
        # assert (max_voxel_coords >= np.amax(voxel_idxs, axis=0)).all(), 'Shape: %s' % (str(voxel_idxs.shape))
        voxel_idxs = voxel_idxs - min_voxel_coords
        voxel_idxs = voxel_idxs[:, 0:4, 0:2]
        x_idxs, y_idxs = voxel_idxs[:, :, 0].copy(), voxel_idxs[:, :, 1].copy()
        voxel_idxs[:, :, 0] = voxelized_shape[1] - y_idxs
        voxel_idxs[:, :, 1] = voxelized_shape[0] - x_idxs
        return voxel_idxs

    def pt_to_bev_pix(self, pt):
        """
        :param corners3d: (N, 3)
        :return:
            bev_corners: (N, 2)
        """
        voxel_size = np.array(self.voxel_size)
        area_scope = np.array(self.area_scope)

        voxel_idxs = np.floor(pt / voxel_size).astype(np.int32)

        min_voxel_coords = np.floor(area_scope[:, 0] / voxel_size).astype(np.int32)
        max_voxel_coords = np.ceil(area_scope[:, 1] / voxel_size - 1).astype(np.int32)
        voxelized_shape = ((max_voxel_coords - min_voxel_coords) + 1).astype(np.int32)

        # Check the points are bounded by the image scope
        # assert (min_voxel_coords <= np.amin(voxel_idxs, axis=0)).all(), 'Shape: %s' % (str(voxel_idxs.shape))
        # assert (max_voxel_coords >= np.amax(voxel_idxs, axis=0)).all(), 'Shape: %s' % (str(voxel_idxs.shape))
        voxel_idxs = voxel_idxs - min_voxel_coords
        voxel_idxs = voxel_idxs[:, 0:2]
        x_idxs, y_idxs = voxel_idxs[:, 0].copy(), voxel_idxs[:, 1].copy()
        voxel_idxs[:, 0] = voxelized_shape[1] - y_idxs
        voxel_idxs[:, 1] = voxelized_shape[0] - x_idxs
        return voxel_idxs

    def draw_bev_circle_scale(self, image, circle_delta=10, circle_range=None, color=(0, 255, 0), thickness=1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        circle_center = np.array([[0., 0., 0.]], dtype=np.float32)
        circle_center = self.pt_to_bev_pix(circle_center)
        center_x, center_y = circle_center[0, 0], circle_center[0, 1]
        if circle_range is None:
            circle_range = abs(np.asarray(self.area_scope)).max()
        for rad in np.arange(circle_delta, circle_delta + circle_range, circle_delta, dtype=np.float32):
            circle_point = np.array([[rad, 0, 0]], dtype=np.float32)
            circle_point = self.pt_to_bev_pix(circle_point)
            circle_point_x, circle_point_y = circle_point[0, 0], circle_point[0, 1]
            circle_radius = np.abs(circle_point_y - center_y)
            circle_text = str(int(rad)) + 'm'
            cv2.circle(image, (center_x, center_y), circle_radius, color, thickness=thickness)
            cv2.putText(image, '%s' % circle_text, (circle_point_x - 10, circle_point_y), font,
                        0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return image

    def draw_bev_det_scope(self, image, det_scope=[[-50, 50], [-50, 50]], color=(0, 0, 255), thickness=1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        det_box = np.array([
            [det_scope[0][0], det_scope[1][0], 0.],
            [det_scope[0][1], det_scope[1][1], 0.],
        ], dtype=np.float32
        )
        det_box = self.pt_to_bev_pix(det_box)
        x1, y1 = det_box[0][0], det_box[0][1]
        x2, y2 = det_box[1][0], det_box[1][1]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        return image

    def draw_bev_polys(self, bev_map, polys, thickness = 2, color = (224, 224, 224)):
        if polys is None or len(polys) == 0:
            return bev_map
        else:
            bev_map = self.draw_poly_lines(image = bev_map, polys = polys, color=color, thickness=thickness)
            return bev_map

    def draw_poly_lines(self, image, polys, color=(0, 255, 0), thickness=2):
        for k in range(len(polys)):
            vertices_array = np.asarray(polys[k], dtype=np.float32).reshape(-1, 3)
            for j in range(vertices_array.shape[0]):
                ori_index = j
                end_index = (j+1) % vertices_array.shape[0]
                line_vertices = np.array([
                    vertices_array[ori_index, :],
                    vertices_array[end_index, :]
                ], dtype = np.float)
                line_pixels = self.pt_to_bev_pix(line_vertices)
                x1, y1 = line_pixels[0, 0], line_pixels[0, 1]
                x2, y2 = line_pixels[1, 0], line_pixels[1, 1]
                cv2.line(image, (x1, y1), (x2, y2), color, thickness=thickness)
        return image

    def draw_bev_boxes(self, bev_map, boxes3d, labels, scores=None, track_ids=None, 
        thickness=2, colorize_with_label = True, color=(0, 255, 0)):
        classes = ['', 'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Cone', 'Unknown', 'Dontcare']
        if boxes3d is None or boxes3d.shape[0] == 0:
            return bev_map
        corners3d = self.boxes3d_to_corners3d_lidar(boxes3d)
        bev_corners = self.corners3d_to_bev_corners(corners3d)

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
                bev_color_map = np.asarray(self.colormap)
                bev_color_map = bev_color_map[:, [2, 1, 0]]
                if scores is not None:
                    bev_map = self.draw_box_lines(bev_map, bev_corners[mask],
                                                  color=tuple(bev_color_map[cur_label]),
                                                  scores=scores[mask], thickness=thickness, track_ids=masked_track_ids,
                                                  box_labels=[classes[cur_label]] * mask.sum())
                else:
                    bev_map = self.draw_box_lines(bev_map, bev_corners[mask],
                                                  color=tuple(bev_color_map[cur_label]),
                                                  scores=None, thickness=thickness, track_ids=masked_track_ids,
                                                  box_labels=[classes[cur_label]] * mask.sum())
            else:
                if scores is not None:
                    bev_map = self.draw_box_lines(bev_map, bev_corners[mask], color=tuple(color),
                                                  scores=scores[mask], thickness=thickness, track_ids=masked_track_ids,
                                                  box_labels=[classes[cur_label]] * mask.sum())
                else:
                    bev_map = self.draw_box_lines(bev_map, bev_corners[mask], color=tuple(color),
                                                  scores=None, thickness=thickness, track_ids=masked_track_ids,
                                                  box_labels=[classes[cur_label]] * mask.sum())
        return bev_map

    def draw_box_lines(self, image, bev_corners, color=(0, 255, 0), scores=None, thickness=2, 
                       track_ids=None, box_labels=None, arrow=True):
        """
        :param image: (H, W)
        :param bev_corners: (N, 4, 2)
        :return:
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = tuple([int(x) for x in color])
        for k in range(bev_corners.shape[0]):
            # draw top boxes
            for j in range(0, 4):
                x1, y1 = bev_corners[k, j, 0], bev_corners[k, j, 1]
                x2, y2 = bev_corners[k, (j + 1) % 4, 0], bev_corners[k, (j + 1) % 4, 1]
                cv2.line(image, (x1, y1), (x2, y2), color, thickness=thickness)
            center = bev_corners[k].mean(axis=0)

            # Draw the type of object
            # cv2.putText(image, '%s' % box_labels[k], (bev_corners[k, 0, 0] - 30, bev_corners[k, 0, 1]), font,
            #             0.5, color, 1, cv2.LINE_AA)

            # Draw the scores of object
            # if scores is not None:
            #   cv2.putText(image, '%.2f' % scores[k], (bev_corners[k, 0, 0], bev_corners[k, 0, 1]), font,
            #               0.5, color, 1, cv2.LINE_AA)

            if track_ids is None:
                head = (bev_corners[k][0] + bev_corners[k][1]) / 2
                head = (head - center) * 1.5 + center
                center = np.round(center).astype(np.int32)
                head = np.round(head).astype(np.int32)
                if(arrow):image = cv2.arrowedLine(image, tuple(center), tuple(head), color, thickness=thickness)
            else:
                # track_ids: id, vx, vy, vz
                obj_id = track_ids[k, 0]
                velocity_vec = np.array((-1.0 * track_ids[k, 2], -1.0 * track_ids[k, 1]), dtype = np.float32)
                head = center + velocity_vec * 3.0
                center = np.round(center).astype(np.int32)
                head = np.round(head).astype(np.int32)
                if(arrow):image = cv2.arrowedLine(image, tuple(center), tuple(head), color, thickness=thickness)
                cv2.putText(image, '%d' % obj_id, (bev_corners[k, 0, 0]-2, bev_corners[k, 0, 1]), font,
                    0.7, (255,255,255), 2, cv2.LINE_AA)
        return image


    def get_part_color_by_offset(self, pts_offset):
        """
        :param pts_offset: (N, 3) offset in xyz, 0~1
        :return:
        """
        color_st = np.array([60, 60, 60], dtype=np.uint8)
        color_ed = np.array([230, 230, 230], dtype=np.uint8)
        pts_color = pts_offset * (color_ed - color_st).astype(np.float32) + color_st.astype(np.float32)
        pts_color = np.clip(np.round(pts_color), a_min=0, a_max=255)
        pts_color = pts_color.astype(np.uint8)
        return pts_color
