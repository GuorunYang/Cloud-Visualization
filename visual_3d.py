import os
import cv2
import math
import time
import warnings
import numpy as np
import data_loader
from tqdm import tqdm

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class Visual3D(object):
    """
        3D Visualization for point cloud and detections
    """
    def __init__(self, voxel_size = None, area_scope = None, colormap = None, viewpoint = None):
        # Default parameters
        self.viewpoint = "vehicle" 
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
        self.score_thresh = [
            0.3, 
            0.35,
            0.35, 
            0.3,
            0.35
        ]
        if not voxel_size is None:
            self.voxel_size = voxel_size
        if not area_scope is None:
            self.area_scope = area_scope
        if not colormap is None:
            self.colormap = colormap
        if not viewpoint is None:
            self.viewpoint = viewpoint


    def visualization(self, draw_status, draw_lists, draw_elements, save_path):
        import mayavi.mlab as mlab
        if not draw_status["debug"]:
            mlab.options.offscreen = True
        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000))
        if draw_status["draw_frame"]:
            frame_cloud_path = draw_lists["cloud_list"][0]
            frame_results, frame_labels, frame_polys = None, None, None
            frame_voxel_path, frame_image_path = None, None
            frame_name = os.path.splitext(os.path.basename(frame_cloud_path))[0]
            if draw_status["draw_result"]:
                frame_results = draw_elements["results"][frame_name]
            if draw_status["draw_label"]:
                frame_labels = draw_elements["labels"][frame_name]
            # if draw_status["draw_poly"]:
            #     frame_polys = draw_elements["polys"][frame_name]
            # if draw_status["draw_voxel"]:
            #     frame_voxel_path = draw_lists["voxel_list"][frame_name]
            # if draw_status["draw_image"]:
            #     frame_image_path = draw_lists["image_list"][frame_name]
            fig = self.draw_3d_map(fig, draw_status, frame_cloud_path, frame_results, 
                frame_labels, frame_polys, frame_voxel_path, frame_image_path)
            if draw_status["debug"]:
                mlab.show()
            # Save the image
            frame_3d_path = (frame_cloud_path.split('/')[-1]).split('.')[0] + '.png'
            if os.path.isdir(save_path):
                # If the directory is provided, the map is writed into the directory
                frame_3d_fn = (frame_cloud_path.split('/')[-1]).rsplit('.', 1)[0] + '.png'
                frame_3d_path = os.path.join(save_path, frame_3d_fn)
            elif save_path.endswith('.png'):
                # If the png path is provided, the map is save as the path
                frame_3d_path = save_path
            else:
                print('Please check the path of save image: {}. Maybe the directory does not exist or the file exists already'.format(frame_bev_path))
            mlab.savefig(frame_3d_path)
            print("Rendering image saves to {}".format(frame_3d_path))
            mlab.clf(figure=fig)
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
                        warnings.warn("Cannot find frame {} in results".format(frame_name))
                if draw_status["draw_label"]:
                    if frame_name in draw_elements["labels"]:
                        frame_labels = draw_elements["labels"][frame_name]
                    else:
                        warnings.warn("Cannot find frame {} in labels".format(frame_name))
                # if draw_status["draw_poly"]:
                #     frame_polys = draw_elements["polys"][i]
                # if draw_status["draw_voxel"]:
                #     frame_voxel_path = draw_lists["voxel_list"][i]
                # if draw_status["draw_image"]:
                #     frame_image_path = draw_lists["image_list"][i]
                fig = self.draw_3d_map(fig, draw_status, frame_cloud_path, frame_results, 
                    frame_labels, frame_polys, frame_voxel_path, frame_image_path)
                if draw_status["debug"]:
                    mlab.show()
                frame_3d_fn = os.path.splitext(frame_cloud_name)[0] + '.png'
                mlab.savefig(os.path.join(save_path, frame_3d_fn))
                mlab.clf(figure=fig)
            print("Rendering image saves to {}".format(save_path))
        mlab.close()
        return True


    def draw_3d_map(self, fig, draw_status, cloud_path, frame_results = None, frame_labels = None, frame_polys = None,
                 frame_voxel_path = None, frame_image_path = None):
        import mayavi.mlab as mlab
        if draw_status["draw_cloud"]:
            cloud = data_loader.load_cloud(cloud_path)
            fig = self.draw_lidar(cloud, fig=fig)
            if draw_status["draw_result"] and draw_status["draw_label"]:
                gt_color    = (178/255.0, 255/255.0, 102/255.0)
                det_color   = (255/255.0, 151/255.0, 45/255.0)
                det_boxes3d, det_scores, det_cls, det_trackids = data_loader.get_boxes_from_results(frame_results)
                gt_boxes3d, gt_scores, gt_cls, gt_trackids = data_loader.get_boxes_from_labels(frame_labels)
                fig = self.draw_3d_boxes(fig, boxes3d=det_boxes3d, labels=det_cls,
                                    scores=det_scores, track_ids=det_trackids,
                                    colorize_with_label = False, color=det_color, thickness = 2)
                fig = self.draw_3d_boxes(fig, boxes3d=gt_boxes3d, labels=gt_cls,
                                    scores=gt_scores, track_ids=gt_trackids,
                                    colorize_with_label = False, color=gt_color, thickness = 2)
            else:
                if draw_status["draw_label"]:
                    gt_boxes3d, gt_scores, gt_cls, gt_trackids = data_loader.get_boxes_from_labels(frame_labels)
                    fig = self.draw_3d_boxes(fig, boxes3d=gt_boxes3d, labels=gt_cls, scores=gt_scores, track_ids=gt_trackids)
                if draw_status["draw_result"]:
                    det_boxes3d, det_scores, det_cls, det_trackids = data_loader.get_boxes_from_results(frame_results)
                    fig = self.draw_3d_boxes(fig, boxes3d=det_boxes3d, labels=det_cls, scores=det_scores, track_ids=det_trackids)
                if draw_status["draw_poly"]:
                    poly_boxes3d, poly_vertices3d, poly_cls, poly_trackids = self.get_polys(frame_polys)
                    fig = self.draw_3d_polys(fig, vertices3d = poly_vertices3d)
                if draw_status["draw_voxel"]:
                    voxel_color = tuple(np.array([216, 216, 216]) / 255)
                    fig = self.draw_3d_voxels(fig, voxel_path = frame_voxel_path, height_offset = -1.4,
                        voxel_height = 1.7, voxel_color = voxel_color)
            if draw_status["draw_ground"]:
                fig = self.draw_ground_plane(fig)

            if self.viewpoint != "v2x":
                # mlab.view(azimuth=179.84, elevation=65.03, distance=95.96,
                #           focalpoint=np.array([13.19, -0.48, 4.50]), roll=89.99)
                if draw_status["draw_voxel"]:
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
            return fig
        else:
            warnings.warn("No cloud..")
            return fig


    def draw_me(self, img, car_vertex, car_path, car_ratio, height_delta = 20):
        car_img = cv2.imread(car_path)
        mask_img = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)
        car_height, car_width = car_img.shape[0], car_img.shape[1]
        car_height = int(car_height * car_ratio)
        car_width = int(car_width * car_ratio)
        car_img = cv2.resize(car_img, (car_width, car_height), interpolation=cv2.INTER_LINEAR)
        mask_img = cv2.resize(mask_img, (car_width, car_height), interpolation=cv2.INTER_NEAREST)

        car_left = car_vertex[0] - int(car_width / 2.0)
        car_right = car_left + car_width
        car_up = car_vertex[1] - int(car_height / 2.0) + height_delta
        car_down = car_up + car_height

        # img_roi = img[car_up:car_down, car_left:car_right, :]
        # bless_roi = cv2.addWeighted(img_roi, 1-0.5, car_img, 0.5, 0.0)
        # img[car_up:car_down, car_left:car_right, :] = bless_roi

        for i in range(car_height - 1):
            for j in range(car_width - 1):
                if mask_img[i, j] != 0:
                    img[car_up + i, car_left+j, :] = car_img[i, j, :]

        # img[car_up:car_down, car_left:car_right, :] = car_img[:, :, :]
        return img


    def draw_3d_polys(self, fig, vertices3d, thickness=3, color=(224, 224, 224)):
        if vertices3d is None or len(vertices3d) == 0:
            return fig
        cur_color = tuple(np.asarray(color, dtype = np.float) / 255.0)
        fig = self.draw_poly3d(vertices3d, fig, poly_color=cur_color, poly_width=thickness)
        return fig


    def draw_3d_boxes(self, fig, boxes3d, labels, scores=None, track_ids=None, 
                      thickness=3, colorize_with_label = True, color=(0, 255, 0)):
        if boxes3d is None or boxes3d.shape[0] == 0:
            return fig
        corners3d = self.boxes3d_to_corners3d_lidar(boxes3d)
        for cur_label in range(labels.min(), labels.max() + 1):
            cur_color = tuple(np.array(self.colormap[cur_label]) / 255)
            # Filter the boxes by score
            if scores is None:
                mask = (labels == cur_label)
            else:
                mask = (labels == cur_label) & (scores > self.score_thresh[cur_label])
            if mask.sum() == 0:
                continue
            # Draw 3D boxes
            if colorize_with_label:
                fig = self.draw_box3d(corners3d[mask], fig, box_color=cur_color, box_width=thickness)
            else:
                fig = self.draw_box3d(corners3d[mask], fig, box_color=tuple(color), box_width=thickness)
            # Draw tracking information
            if track_ids is not None:
                masked_boxes3d = boxes3d[mask]
                masked_corners3d = corners3d[mask]
                masked_track_ids = track_ids[mask]
                fig = self.draw_tracksids(masked_boxes3d, masked_corners3d, masked_track_ids, fig,
                                                        arrow_color=cur_color, arrow_width=thickness)
        return fig


    def draw_legend(self, img, legend_vertex, legend_path, legend_ratio, alpha = 0.7):
        '''
            Draw legend on the image to indicate the category of detections
        '''
        import cv2
        legend_img = cv2.imread(legend_path)
        # legend_img = cv2.cvtColor(legend_img, cv2.COLOR_BGR2RGB)
        legend_height, legend_width = legend_img.shape[0], legend_img.shape[1]
        legend_height = int(legend_height * legend_ratio)
        legend_width = int(legend_width * legend_ratio)
        legend_img = cv2.resize(legend_img, (legend_width, legend_height), interpolation=cv2.INTER_LINEAR)

        legend_left = legend_vertex[0]
        legend_right = legend_left + legend_width
        legend_up = legend_vertex[1]
        legend_down = legend_up + legend_height

        img_roi = img[legend_up:legend_down, legend_left:legend_right, :]
        bless_roi = cv2.addWeighted(img_roi, 1-alpha, legend_img, alpha, 0.0)
        img[legend_up:legend_down, legend_left:legend_right, :] = bless_roi
        return img


    def draw_counts(self, img, count_number, count_color, count_vertex = (100, 100), unit_height = 30, font_size = 14):
        from PIL import ImageFont, ImageDraw, Image
        x, y = count_vertex[0], count_vertex[1]
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype('PingFang-SC-Regular.ttf', font_size)
        for i, cnt in enumerate(count_number):
            y += unit_height
            draw.text((x, y), str(cnt), font=font, fill=tuple(count_color[i]))
        img = np.array(pil_img)
        return img


    def draw_lidar(self, pc, fig=None, pts_scalar=None,
                   bgcolor = (0, 0, 0), pts_color=(85 / 255.0, 197 / 255.0, 228 / 255.0),
                   pts_mode = 'point', scale_factor = 0.5,
                   draw_origin=False, draw_axis=False):
        """ Draw lidar points
            Args:
                pc: numpy array (n,3) of XYZ
                fig: mayavi figure handler, if None create new one otherwise will use it
            Returns:
                fig: created or used fig
        """
        import mayavi.mlab as mlab
        if fig is None:
            fig = mlab.figure(
                figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000)
            )

        mlab.points3d(
            pc[:, 0],
            pc[:, 1],
            pc[:, 2],
            color=pts_color,
            mode=pts_mode,
            colormap="Blues",
            scale_factor=scale_factor,
            figure=fig,
        )

        # draw origin
        if draw_origin:
            mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2)

        # draw axis
        if draw_axis:
            # Define the xyz axis
            axes = np.array(
                [[2.0, 0.0, 0.0, 0.0],
                 [0.0, 2.0, 0.0, 0.0],
                 [0.0, 0.0, 2.0, 0.0]], dtype=np.float64)
            # Draw X-axis
            mlab.plot3d(
                [0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]],
                color=(1, 0, 0), tube_radius=None, figure=fig
            )
            # Draw Y-axis
            mlab.plot3d(
                [0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]],
                color=(0, 1, 0), tube_radius=None, figure=fig
            )
            # Draw Z-axis
            mlab.plot3d(
                [0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]],
                color=(0, 0, 1), tube_radius=None, figure=fig
            )
        return fig

    def voxel2coor(self, voxel_row, voxel_col, coor_scale=0.16, center_pixel = [532, 320],
                   coor_min_x=-63.44, coor_max_x=85.04, coor_min_y=-51.28, coor_max_y=51.28):
        voxel_i = (voxel_row - center_pixel[0]) * -1.0
        voxel_j = (voxel_col - center_pixel[1])
        coor_x = voxel_i * coor_scale
        coor_y = voxel_j * coor_scale
        return coor_x, coor_y


    def draw_3d_voxels(self, fig, voxel_path, voxel_size = 0.60, voxel_height = 1.50, height_offset = 0.0, voxel_color=(0.7, 0.7, 0.7)):
        '''
            Old Version:
                map_dict = {
                    [0, 0, 0]       : 0,    # None
                    [0, 0, 255]     : 1,    # Car
                    [245, 150, 100] : 2,    # Cyclist
                    [145, 230, 100] : 3,    # Truck
                    [250, 80, 100]  : 4,    # Pedestrian
                    [150, 60, 30]   : 5     # Others
                }
            New Version:
                map_dict = {
                    [0, 63, 0]:     : 0,    # None
                    [250, 63, 0]    : 5     # Others
                }
        '''


        # Get the voxels from voxel map
        voxel_image = cv2.imread(voxel_path, cv2.IMREAD_COLOR)
        # other_index = np.where(voxel_image[:, :, 0] == 250)
        other_index = np.where(voxel_image == [150, 60, 30])
        index_array = np.asarray(other_index).transpose()[:, 0:2]
        voxel_boxes3d = np.zeros([index_array.shape[0], 8, 3], dtype = np.float)
        for i in range(index_array.shape[0]):
            voxel_row, voxel_col = index_array[i][0], index_array[i][1]
            coor_x, coor_y = self.voxel2coor(voxel_row, voxel_col)
            # ground_height = float(voxel_image[voxel_row, voxel_col, 1]) / 255.0 * 20 + height_offset
            ground_height = height_offset
            voxel_vertices = np.array([
                [coor_x + voxel_size / 2.0, coor_y - voxel_size / 2.0, ground_height],
                [coor_x - voxel_size / 2.0, coor_y - voxel_size / 2.0, ground_height],
                [coor_x - voxel_size / 2.0, coor_y + voxel_size / 2.0, ground_height],
                [coor_x + voxel_size / 2.0, coor_y + voxel_size / 2.0, ground_height],
                [coor_x + voxel_size / 2.0, coor_y - voxel_size / 2.0, ground_height+voxel_height],
                [coor_x - voxel_size / 2.0, coor_y - voxel_size / 2.0, ground_height+voxel_height],
                [coor_x - voxel_size / 2.0, coor_y + voxel_size / 2.0, ground_height+voxel_height],
                [coor_x + voxel_size / 2.0, coor_y + voxel_size / 2.0, ground_height+voxel_height]
            ], dtype = np.float)
            voxel_boxes3d[i, :, :] = voxel_vertices
        fig = self.mesh_voxel3d(voxel_boxes3d, fig, mesh_opacity = 0.80, max_num=index_array.shape[0])
        return fig

    def draw_sphere_pts(self, pts, color=(0, 1, 0), fig=None, bgcolor=(0, 0, 0), scale_factor=0.2):
        '''
            This function 
        '''
        import mayavi.mlab as mlab
        if fig is None:
            fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(600, 600))

        if isinstance(color, np.ndarray) and color.shape[0] == 1:
            color = color[0]
            color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

        if isinstance(color, np.ndarray):
            pts_color = np.zeros((pts.__len__(), 4), dtype=np.uint8)
            pts_color[:, 0:3] = color
            pts_color[:, 3] = 255
            G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], np.arange(0, pts_color.__len__()), mode='sphere',
                              scale_factor=scale_factor, figure=fig)
            G.glyph.color_mode = 'color_by_scalar'
            G.glyph.scale_mode = 'scale_by_vector'
            G.module_manager.scalar_lut_manager.lut.table = pts_color
        else:
            mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='sphere', color=color,
                          colormap='gnuplot', scale_factor=scale_factor, figure=fig)

        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), line_width=3, tube_radius=None, figure=fig)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), line_width=3, tube_radius=None, figure=fig)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), line_width=3, tube_radius=None, figure=fig)
        return fig

    def proj_3d_to_2d(self, points, fig):
        import mlab_3D_to_2D as mlab_trans
        # Input: n*3
        W = np.ones(points.shape[0])
        hmgns_world_coords = np.column_stack((points, W.transpose()))
        # applying the first transform will give us 'unnormalized' view
        # coordinates we also have to get the transform matrix for the
        # current scene view
        comb_trans_mat = mlab_trans.get_world_to_view_matrix(fig.scene)
        view_coords = mlab_trans.apply_transform_to_points(hmgns_world_coords, comb_trans_mat)

        # to get normalized view coordinates, we divide through by the fourth
        # element
        norm_view_coords = view_coords / (view_coords[:, 3].reshape(-1, 1))

        # the last step is to transform from normalized view coordinates to
        # display coordinates.
        view_to_disp_mat = mlab_trans.get_view_to_display_matrix(fig.scene)
        pixels2d = mlab_trans.apply_transform_to_points(norm_view_coords, view_to_disp_mat)
        pixels2d = pixels2d[:, 0:2]
        return pixels2d

    def get_vertices2d(self, corners3d, fig):
        import mlab_3D_to_2D as mlab_trans
        # Get the top center of 3D box
        top_centers_X = np.zeros((len(corners3d)), dtype=np.float32)
        top_centers_Y = np.zeros((len(corners3d)), dtype=np.float32)
        top_centers_Z = np.zeros((len(corners3d)), dtype=np.float32)
        for n in range(len(corners3d)):
            box_corners3d = corners3d[n]
            top_corners_X = np.array([box_corners3d[4, 0], box_corners3d[5, 0], box_corners3d[6, 0], box_corners3d[7, 0]],
                                     dtype=np.float32)
            top_corners_Y = np.array([box_corners3d[4, 1], box_corners3d[5, 1], box_corners3d[6, 1], box_corners3d[7, 1]],
                                     dtype=np.float32)
            top_corners_Z = np.array([box_corners3d[4, 2], box_corners3d[5, 2], box_corners3d[6, 2], box_corners3d[7, 2]],
                                     dtype=np.float32)
            top_centers_X[n] = np.mean(top_corners_X)
            top_centers_Y[n] = np.mean(top_corners_Y)
            top_centers_Z[n] = np.mean(top_corners_Z)
        # top_centers_X[n] = box_corners3d[6, 0]
        # top_centers_Y[n] = box_corners3d[6, 1]
        # top_centers_Z[n] = box_corners3d[6, 2]

        # Transform the top 3D centers to 2D vertices
        W = np.ones(top_centers_X.shape)
        hmgns_world_coords = np.column_stack((top_centers_X, top_centers_Y, top_centers_Z, W))

        # applying the first transform will give us 'unnormalized' view
        # coordinates we also have to get the transform matrix for the
        # current scene view
        comb_trans_mat = mlab_trans.get_world_to_view_matrix(fig.scene)
        view_coords = mlab_trans.apply_transform_to_points(hmgns_world_coords, comb_trans_mat)

        # to get normalized view coordinates, we divide through by the fourth
        # element
        norm_view_coords = view_coords / (view_coords[:, 3].reshape(-1, 1))

        # the last step is to transform from normalized view coordinates to
        # display coordinates.
        view_to_disp_mat = mlab_trans.get_view_to_display_matrix(fig.scene)
        vertices2d = mlab_trans.apply_transform_to_points(norm_view_coords, view_to_disp_mat)
        vertices2d = vertices2d[:, 0:2]
        return vertices2d

    def get_velocity_origins(self, boxes3d, corners3d, max_num=500):
        num = min(max_num, len(corners3d))
        origin_array = np.zeros((num, 3), dtype = np.float32)
        for n in range(num):
            # Accumulate the lines
            b = corners3d[n]  # (8, 3)
            i, j = 0, 5
            x_mean = (b[i, 0] + b[j, 0]) / 2.0
            y_mean = (b[i, 1] + b[j, 1]) / 2.0
            z_mean = (b[i, 2] + b[j, 2]) / 2.0
            origin_array[n, :] = [x_mean, y_mean, z_mean]
            # x_center, y_center, z_center = boxes3d[n, 0], boxes3d[n, 1], boxes3d[n, 2]
            # z_center += boxes3d[n, 5] / 2.0
            # origin_array[n, :] = [x_center, y_center, z_center]
        return origin_array

    def accumulate_plane_triangulars(self, corners3d, max_num=500):
        if corners3d.shape[0] > max_num:
            corners3d = corners3d[0:max_num, :, :]
        plane_triangles_array = np.zeros([6, len(corners3d) * 2, 3], dtype = np.int)
        vertices_array = corners3d.reshape(len(corners3d)*8, 3)
        for n in range(len(corners3d)):
            vertex_index = n * 8
            triangle_index = n * 2
            plane_triangles_array[0, triangle_index+0, :] = np.array([vertex_index, vertex_index+2, vertex_index+1], dtype = np.int)
            plane_triangles_array[0, triangle_index+1, :] = np.array([vertex_index, vertex_index+2, vertex_index+3], dtype = np.int)
            plane_triangles_array[3, triangle_index+0, :] = np.array([vertex_index+6, vertex_index+4, vertex_index+5], dtype = np.int)
            plane_triangles_array[3, triangle_index+1, :] = np.array([vertex_index+6, vertex_index+4, vertex_index+7], dtype = np.int)
            plane_triangles_array[1, triangle_index+0, :] = np.array([vertex_index, vertex_index+5, vertex_index+1], dtype = np.int)
            plane_triangles_array[1, triangle_index+1, :] = np.array([vertex_index, vertex_index+5, vertex_index+4], dtype = np.int)
            plane_triangles_array[4, triangle_index+0, :] = np.array([vertex_index+6, vertex_index+3, vertex_index+2], dtype = np.int)
            plane_triangles_array[4, triangle_index+1, :] = np.array([vertex_index+6, vertex_index+3, vertex_index+7], dtype = np.int)
            plane_triangles_array[2, triangle_index+0, :] = np.array([vertex_index, vertex_index+7, vertex_index+3], dtype = np.int)
            plane_triangles_array[2, triangle_index+1, :] = np.array([vertex_index, vertex_index+7, vertex_index+4], dtype = np.int)
            plane_triangles_array[5, triangle_index+0, :] = np.array([vertex_index+6, vertex_index+1, vertex_index+2], dtype = np.int)
            plane_triangles_array[5, triangle_index+1, :] = np.array([vertex_index+6, vertex_index+1, vertex_index+5], dtype = np.int)
        return vertices_array, plane_triangles_array

    def accumulate_triangulars(self, corners3d, max_num=500):
        """
            This function accumulates the triangular meshes for visualization.
            Specifically, for each plane, we concat two triangular meshes to visualize.
                7 -------- 4
               /|         /|
              6 -------- 5 .
              | |        | |
              . 3 -------- 0
              |/         |/
              2 -------- 1
            Args:
                corners3d: (n, 8, 3)
            Returns:
                vertices_array: (n*8, 3)
                triangles_array: (n)
        """
        if corners3d.shape[0] > max_num:
            corners3d = corners3d[0:max_num, :, :]
        triangles_array = np.zeros([len(corners3d) * 12, 3], dtype = np.int)
        vertices_array = corners3d.reshape(len(corners3d)*8, 3)
        for n in range(len(corners3d)):
            vertex_index = n * 8
            triangle_index = n * 12
            triangles_array[triangle_index+0, :] = np.array([vertex_index, vertex_index+2, vertex_index+1], dtype = np.int)
            triangles_array[triangle_index+1, :] = np.array([vertex_index, vertex_index+2, vertex_index+3], dtype = np.int)
            triangles_array[triangle_index+2, :] = np.array([vertex_index, vertex_index+5, vertex_index+1], dtype = np.int)
            triangles_array[triangle_index+3, :] = np.array([vertex_index, vertex_index+5, vertex_index+4], dtype = np.int)
            triangles_array[triangle_index+4, :] = np.array([vertex_index, vertex_index+7, vertex_index+3], dtype = np.int)
            triangles_array[triangle_index+5, :] = np.array([vertex_index, vertex_index+7, vertex_index+4], dtype = np.int)
            triangles_array[triangle_index+6, :] = np.array([vertex_index+6, vertex_index+1, vertex_index+2], dtype = np.int)
            triangles_array[triangle_index+7, :] = np.array([vertex_index+6, vertex_index+1, vertex_index+5], dtype = np.int)
            triangles_array[triangle_index+8, :] = np.array([vertex_index+6, vertex_index+3, vertex_index+2], dtype = np.int)
            triangles_array[triangle_index+9, :] = np.array([vertex_index+6, vertex_index+3, vertex_index+7], dtype = np.int)
            triangles_array[triangle_index+10, :] = np.array([vertex_index+6, vertex_index+4, vertex_index+5], dtype = np.int)
            triangles_array[triangle_index+11, :] = np.array([vertex_index+6, vertex_index+4, vertex_index+7], dtype = np.int)
        return vertices_array, triangles_array

    def accumulate_box3d_lines(self, corners3d, max_num=500):
        """
            This function accumulates the triangular meshes for visualization.
            Specifically, for each plane, we concat two triangular meshes to visualize.
                7 -------- 4
               /|         /|
              6 -------- 5 .
              | |        | |
              . 3 -------- 0
              |/         |/
              2 -------- 1
            Args:
                corners3d: (n, 8, 3)
            Returns:
                line_vertices: (n*28, 3)
                line_connections: (n*14, )
        """
        if corners3d.shape[0] > max_num:
            corners3d = corners3d[0:max_num, :, :]
        box_vertices = corners3d.reshape(len(corners3d)*8, 3)
        line_vertices = np.zeros([len(corners3d)*28, 3], dtype = np.float)
        for n in range(len(corners3d)):
            # Accumulate the lines
            box_vertex_index = n * 8
            line_vertex_index = n * 14 * 2
            line_vertices[line_vertex_index+0, :] = box_vertices[box_vertex_index+0, :]     # Vertex: 0->1
            line_vertices[line_vertex_index+1, :] = box_vertices[box_vertex_index+1, :]
            line_vertices[line_vertex_index+2, :] = box_vertices[box_vertex_index+0, :]     # Vertex: 0->3
            line_vertices[line_vertex_index+3, :] = box_vertices[box_vertex_index+3, :]
            line_vertices[line_vertex_index+4, :] = box_vertices[box_vertex_index+0, :]     # Vertex: 0->4
            line_vertices[line_vertex_index+5, :] = box_vertices[box_vertex_index+4, :]
            line_vertices[line_vertex_index+6, :] = box_vertices[box_vertex_index+2, :]     # Vertex: 2->1
            line_vertices[line_vertex_index+7, :] = box_vertices[box_vertex_index+1, :]
            line_vertices[line_vertex_index+8, :] = box_vertices[box_vertex_index+2, :]     # Vertex: 2->3
            line_vertices[line_vertex_index+9, :] = box_vertices[box_vertex_index+3, :]
            line_vertices[line_vertex_index+10, :] = box_vertices[box_vertex_index+2, :]     # Vertex: 2->6
            line_vertices[line_vertex_index+11, :] = box_vertices[box_vertex_index+6, :]
            line_vertices[line_vertex_index+12, :] = box_vertices[box_vertex_index+5, :]     # Vertex: 5->1
            line_vertices[line_vertex_index+13, :] = box_vertices[box_vertex_index+1, :]
            line_vertices[line_vertex_index+14, :] = box_vertices[box_vertex_index+5, :]     # Vertex: 5->4
            line_vertices[line_vertex_index+15, :] = box_vertices[box_vertex_index+4, :]
            line_vertices[line_vertex_index+16, :] = box_vertices[box_vertex_index+5, :]     # Vertex: 5->6
            line_vertices[line_vertex_index+17, :] = box_vertices[box_vertex_index+6, :]
            line_vertices[line_vertex_index+18, :] = box_vertices[box_vertex_index+7, :]     # Vertex: 7->3
            line_vertices[line_vertex_index+19, :] = box_vertices[box_vertex_index+3, :]
            line_vertices[line_vertex_index+20, :] = box_vertices[box_vertex_index+7, :]     # Vertex: 7->4
            line_vertices[line_vertex_index+21, :] = box_vertices[box_vertex_index+4, :]
            line_vertices[line_vertex_index+22, :] = box_vertices[box_vertex_index+7, :]     # Vertex: 7->6
            line_vertices[line_vertex_index+23, :] = box_vertices[box_vertex_index+6, :]
            line_vertices[line_vertex_index+24, :] = box_vertices[box_vertex_index+0, :]     # Vertex: 0->5
            line_vertices[line_vertex_index+25, :] = box_vertices[box_vertex_index+5, :]
            line_vertices[line_vertex_index+26, :] = box_vertices[box_vertex_index+1, :]     # Vertex: 1->4
            line_vertices[line_vertex_index+27, :] = box_vertices[box_vertex_index+4, :]
        line_connections = np.arange(0, line_vertices.shape[0]).reshape(-1, 2)
        return line_vertices, line_connections

    def accmulate_poly3d_lines(self, vertices3d):
        '''
            For each 3D polygon, the vertices3d contains the bottom vertices and top vertices,
            so that we successively connect vertices.
        '''
        line_vertices = np.empty([0, 3], dtype=np.float)
        for k in range(len(vertices3d)):
            vertices_array = np.asarray(vertices3d[k], dtype=np.float32).reshape(-1, 3)
            slice_index = int(len(vertices_array) / 2)
            top_vertices_array = vertices_array[0:slice_index, :]
            bot_vertices_array = vertices_array[slice_index:, :]
            for j in range(top_vertices_array.shape[0]):
                ori_index = j
                end_index = (j+1) % top_vertices_array.shape[0]
                line_vertices = np.vstack((line_vertices, top_vertices_array[ori_index, :]))
                line_vertices = np.vstack((line_vertices, top_vertices_array[end_index, :]))
                line_vertices = np.vstack((line_vertices, bot_vertices_array[ori_index, :]))
                line_vertices = np.vstack((line_vertices, bot_vertices_array[end_index, :]))
                line_vertices = np.vstack((line_vertices, top_vertices_array[ori_index, :]))
                line_vertices = np.vstack((line_vertices, bot_vertices_array[ori_index, :]))
        line_connections = np.arange(0, line_vertices.shape[0]).reshape(-1, 2)
        return line_vertices, line_connections

    def draw_poly3d(self, vertices3d, fig, poly_color=(1, 1, 1), poly_width=1):
        import mayavi.mlab as mlab
        line_vertices, line_connections = self.accmulate_poly3d_lines(vertices3d)
        src = mlab.pipeline.scalar_scatter(line_vertices[:, 0], line_vertices[:, 1], line_vertices[:, 2])
        src.mlab_source.dataset.lines = line_connections
        src.update()
        lines = mlab.pipeline.stripper(src)
        mlab.pipeline.surface(lines, color=poly_color, line_width=poly_width, opacity=1.0, figure=fig)
        return fig

    def draw_box3d(self, corners3d, fig, box_color=(1, 1, 1), box_width=1, max_num=500):
        import mayavi.mlab as mlab
        line_vertices, line_connections = self.accumulate_box3d_lines(corners3d, max_num)
        src = mlab.pipeline.scalar_scatter(line_vertices[:, 0], line_vertices[:, 1], line_vertices[:, 2])
        src.mlab_source.dataset.lines = line_connections
        src.update()
        lines = mlab.pipeline.stripper(src)
        mlab.pipeline.surface(lines, color=box_color, line_width=box_width, opacity=1.0, figure=fig)
        self.mesh_box3d(corners3d = corners3d, fig = fig, mesh_color = box_color, mesh_opacity = 0.35,
                   lighting = False, max_num = max_num)
        return fig

    def mesh_voxel3d(self, corners3d, fig, mesh_opacity = 0.80, max_num=500):
        import mayavi.mlab as mlab
        vertices_array, plane_triangles_array = self.accumulate_plane_triangulars(corners3d, max_num)
        for k in range(plane_triangles_array.shape[0]):
            voxel_color = tuple(np.array([216, 216, 216]) / 255)
            # if k % 3 == 0:
            #   voxel_color = tuple(np.array([216, 216, 216]) / 255)
            # elif k % 3 == 1:
            #   voxel_color = tuple(np.array([121, 121, 121]) / 255)
            # elif k % 3 == 2:
            #   voxel_color = tuple(np.array([171, 171, 171]) / 255)
            mesh_out = mlab.triangular_mesh(vertices_array[:, 0], vertices_array[:, 1], vertices_array[:, 2],
                plane_triangles_array[k], color=voxel_color, opacity=mesh_opacity, figure=fig)
            mesh_out.actor.property.lighting = True
        return fig

    def mesh_box3d(self, corners3d, fig, mesh_color=(1, 1, 1), mesh_opacity = 0.35, lighting = False, max_num=500):
        import mayavi.mlab as mlab
        vertices_array, triangles_array = self.accumulate_triangulars(corners3d, max_num)
        mesh_out = mlab.triangular_mesh(vertices_array[:, 0], vertices_array[:, 1], vertices_array[:, 2], triangles_array,
                             color=mesh_color, opacity = mesh_opacity, figure=fig)
        mesh_out.actor.property.lighting = lighting
        return fig

    def draw_tracksids(self, track_boxes3d, track_corners3d, track_ids, fig, arrow_color=(1, 1, 1), arrow_width = 3):
        import mayavi.mlab as mlab
        # track_boxes3d: (N, 7) [x, y, z, w, l, h, ry]
        # track_ids: (N, 4) [id, vx, vy, vz]
        track_origins3d = self.get_velocity_origins(track_boxes3d,track_corners3d)
        for k in range(track_ids.shape[0]):
            mlab.quiver3d(track_origins3d[k, 0], track_origins3d[k, 1], track_origins3d[k, 2],
                          track_ids[k, 1], track_ids[k, 2], track_ids[k, 3],
                          scale_factor = 0.01, mode = 'arrow',
                          color=arrow_color, line_width = 2.0, figure = fig)
            mlab.text3d(track_boxes3d[k, 0], track_boxes3d[k, 1], track_boxes3d[k, 2] + track_boxes3d[k, 5] * 1.5,
                        text=str(int(track_ids[k, 0])),
                        line_width = arrow_width, color = arrow_color, figure = fig)
        return fig

    def draw_corners3d(self, corners3d, fig, color=(1, 1, 1), line_width=1, cls=None, tag='', max_num=500, tube_radius=None):
        """
        :param corners3d: (N, 8, 3)
        :param fig:
        :param color:
        :param line_width:
        :param cls:
        :param tag:
        :param max_num:
        :return:
        """
        import mayavi.mlab as mlab
        num = min(max_num, len(corners3d))
        for n in range(num):
            b = corners3d[n]  # (8, 3)
            if cls is not None:
                if isinstance(cls, np.ndarray):
                    mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%.2f' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)
                else:
                    mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%s' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)

            for k in range(0, 4):
                i, j = k, (k + 1) % 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                            tube_radius=tube_radius,
                            line_width=line_width, figure=fig)

                i, j = k + 4, (k + 1) % 4 + 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                            tube_radius=tube_radius,
                            line_width=line_width, figure=fig)

                i, j = k, k + 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                            tube_radius=tube_radius,
                            line_width=line_width, figure=fig)

            i, j = 0, 5
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)
            i, j = 1, 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        return fig

    def draw_grid(self, x1, y1, x2, y2, fig, tube_radius=0.05, color=(0.5, 0.5, 0.5)):
        from mayavi import mlab
        mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=None, line_width=1, figure=fig)

        # mlab.plot3d([x1, x1], [0, 0], [z1, z2], color=color, tube_radius=tube_r, line_width=1, figure=fig)
        # mlab.plot3d([x2, x2], [0, 0], [z1, z2], color=color, tube_radius=tube_r, line_width=1, figure=fig)
        # mlab.plot3d([x1, x2], [0, 0], [z1, z1], color=color, tube_radius=tube_r, line_width=1, figure=fig)
        # mlab.plot3d([x1, x2], [0, 0], [z2, z2], color=color, tube_radius=tube_r, line_width=1, figure=fig)
        return fig

    def draw_multi_grid_range(self, fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
        # import pdb
        # pdb.set_trace()
        for x in range(bv_range[0], bv_range[2], grid_size):
            for y in range(bv_range[1], bv_range[3], grid_size):
                fig = self.draw_grid(x, y, x + grid_size, y + grid_size, fig, tube_radius=0.02)

        return fig

    def draw_multi_grid(self, fig):
        block_area = np.array([[0, -30, -5, 20, -10, 5],
                               [0, -10, -5, 20, 10, 5],
                               [0, 10, -5, 20, 30, 5],
                               [20, -30, -5, 40, -10, 5],
                               [20, -10, -5, 40, 10, 5],
                               [20, 10, -5, 40, 30, 5],
                               [40, -30, -5, 60, -10, 5],
                               [40, -10, -5, 60, 10, 5],
                               [40, 10, -5, 60, 30, 5],
                               [60, -30, -5, 80, -10, 5],
                               [60, -10, -5, 80, 10, 5],
                               [60, 10, -5, 80, 30, 5]], dtype=np.float32)
        for k in range(block_area.__len__()):
            fig = self.draw_grid(block_area[k, 0], block_area[k, 1], block_area[k, 3], block_area[k, 4], fig)
        return fig

    def draw_plane(self, fig, plane, color=(0.8, 0.5, 0)):
        """
        :param fig:
        :param plane: (a, b, c, d)
        :return:
        """
        import mayavi.mlab as mlab
        a, b, c, d = plane

        x_idxs = np.arange(-30, 30, 1).astype(np.float32)
        z_idxs = np.arange(0, 70, 1).astype(np.float32)

        x_idxs, z_idxs = np.meshgrid(x_idxs, z_idxs)
        x_idxs, z_idxs = x_idxs.reshape(-1), z_idxs.reshape(-1)
        y_idxs = (-d - a * x_idxs - c * z_idxs) / b

        mlab.points3d(x_idxs, y_idxs, z_idxs, mode='sphere', color=color,
                      colormap='gnuplot', scale_factor=0.1, figure=fig)
        return fig

    def draw_ground_plane(self, fig, ground_height = -1.60, ground_color=(0.5, 0.5, 0.5)):
        import mayavi.mlab as mlab
        x_idxs, y_idxs = np.mgrid[-70.:70.:0.1, -30.:30.:0.1]
        z_idxs = np.full_like(x_idxs, ground_height, dtype=np.float32)
        mlab.surf(x_idxs, y_idxs, z_idxs, color = ground_color, colormap='gnuplot', figure = fig)
        return fig

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