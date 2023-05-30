import numpy as np
import warnings
import cv2
import math

def draw_lidar(pc, fig=None, pts_scalar=None,
               bgcolor = (0, 0, 0), pts_color=(85 / 255.0, 197 / 255.0, 228 / 255.0),
               pts_mode = 'point', scale_factor = 0.5,
               color_by_intensity = False, color_by_ring = False, color_by_label = False,
               draw_origin=False, draw_axis=False
    ):
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
	if color_by_intensity or color_by_ring or color_by_label:
		warnings.warn('The codebase does not support colorize by intensity, ring or labels')

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


def draw_3d_voxels(fig, voxel_path, voxel_size = 0.60, voxel_height = 1.50,
                   height_offset = 0.0, voxel_color=(0.7, 0.7, 0.7)):
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
	def voxel2coor(voxel_row, voxel_col, coor_scale=0.16, coor_min_x=-63.44, coor_max_x=85.04,
	               coor_min_y=-51.28, coor_max_y=51.28, center_pixel = [532, 320]):
		voxel_i = (voxel_row - center_pixel[0]) * -1.0
		voxel_j = (voxel_col - center_pixel[1])
		coor_x = voxel_i * coor_scale
		coor_y = voxel_j * coor_scale
		return coor_x, coor_y

	# Get the voxels from voxel map
	voxel_image = cv2.imread(voxel_path, cv2.IMREAD_COLOR)
	# other_index = np.where(voxel_image[:, :, 0] == 250)
	other_index = np.where(voxel_image == [150, 60, 30])
	index_array = np.asarray(other_index).transpose()[:, 0:2]
	voxel_boxes3d = np.zeros([index_array.shape[0], 8, 3], dtype = np.float)
	for i in range(index_array.shape[0]):
		voxel_row, voxel_col = index_array[i][0], index_array[i][1]
		coor_x, coor_y = voxel2coor(voxel_row, voxel_col)
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

	fig = mesh_voxel3d(voxel_boxes3d, fig, mesh_opacity = 0.80, max_num=index_array.shape[0])
	return fig


def draw_sphere_pts(pts, color=(0, 1, 0), fig=None, bgcolor=(0, 0, 0), scale_factor=0.2):
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


def draw_legend(img, legend_vertex, legend_path, legend_ratio, alpha=0.7):
	legend_img = cv2.imread(legend_path)
	legend_img = cv2.cvtColor(legend_img, cv2.COLOR_BGR2RGB)
	legend_height, legend_width = legend_img.shape[0], legend_img.shape[1]
	legend_height = int(legend_height * legend_ratio)
	legend_width = int(legend_width * legend_ratio)
	legend_img = cv2.resize(legend_img, (legend_width, legend_height), interpolation=cv2.INTER_LINEAR)

	legend_left = legend_vertex[0]
	legend_right = legend_left + legend_width
	legend_up = legend_vertex[1]
	legend_down = legend_up + legend_height

	img_roi = img[legend_up:legend_down, legend_left:legend_right, :]
	bless_roi = cv2.addWeighted(img_roi, 1 - alpha, legend_img, alpha, 0.0)
	img[legend_up:legend_down, legend_left:legend_right, :] = bless_roi
	return img


def proj_3d_to_2d(points, fig):
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


def get_vertices2d(corners3d, fig):
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


def get_velocity_origins(boxes3d, corners3d, max_num=500):
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


def accumulate_plane_triangulars(corners3d, max_num=500):
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


def accumulate_triangulars(corners3d, max_num=500):
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


def accumulate_box3d_lines(corners3d, max_num=500):
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


def accmulate_poly3d_lines(vertices3d):
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


def draw_poly3d(vertices3d, fig, poly_color=(1, 1, 1), poly_width=1):
	import mayavi.mlab as mlab
	line_vertices, line_connections = accmulate_poly3d_lines(vertices3d)
	src = mlab.pipeline.scalar_scatter(line_vertices[:, 0], line_vertices[:, 1], line_vertices[:, 2])
	src.mlab_source.dataset.lines = line_connections
	src.update()
	lines = mlab.pipeline.stripper(src)
	mlab.pipeline.surface(lines, color=poly_color, line_width=poly_width, opacity=1.0, figure=fig)
	return fig


def draw_box3d(corners3d, fig, box_color=(1, 1, 1), box_width=1, max_num=500):
	import mayavi.mlab as mlab
	line_vertices, line_connections = accumulate_box3d_lines(corners3d, max_num)
	src = mlab.pipeline.scalar_scatter(line_vertices[:, 0], line_vertices[:, 1], line_vertices[:, 2])
	src.mlab_source.dataset.lines = line_connections
	src.update()
	lines = mlab.pipeline.stripper(src)
	mlab.pipeline.surface(lines, color=box_color, line_width=box_width, opacity=1.0, figure=fig)
	mesh_box3d(corners3d = corners3d, fig = fig, mesh_color = box_color, mesh_opacity = 0.35,
	           lighting = False, max_num = max_num)
	return fig


def mesh_voxel3d(corners3d, fig, mesh_opacity = 0.80, max_num=500):
	import mayavi.mlab as mlab
	vertices_array, plane_triangles_array = accumulate_plane_triangulars(corners3d, max_num)
	for k in range(plane_triangles_array.shape[0]):
		voxel_color = tuple(np.array([216, 216, 216]) / 255)
		# if k % 3 == 0:
		# 	voxel_color = tuple(np.array([216, 216, 216]) / 255)
		# elif k % 3 == 1:
		# 	voxel_color = tuple(np.array([121, 121, 121]) / 255)
		# elif k % 3 == 2:
		# 	voxel_color = tuple(np.array([171, 171, 171]) / 255)
		mesh_out = mlab.triangular_mesh(vertices_array[:, 0], vertices_array[:, 1], vertices_array[:, 2],
			plane_triangles_array[k], color=voxel_color, opacity=mesh_opacity, figure=fig)
		mesh_out.actor.property.lighting = True
	return fig


def mesh_box3d(corners3d, fig, mesh_color=(1, 1, 1), mesh_opacity = 0.35, lighting = False, max_num=500):
	import mayavi.mlab as mlab
	vertices_array, triangles_array = accumulate_triangulars(corners3d, max_num)
	mesh_out = mlab.triangular_mesh(vertices_array[:, 0], vertices_array[:, 1], vertices_array[:, 2], triangles_array,
	                     color=mesh_color, opacity = mesh_opacity, figure=fig)
	mesh_out.actor.property.lighting = lighting
	return fig


def draw_tracksids(track_boxes3d, track_corners3d, track_ids, fig, arrow_color=(1, 1, 1), arrow_width = 3):
	import mayavi.mlab as mlab
	# track_boxes3d: (N, 7) [x, y, z, w, l, h, ry]
	# track_ids: (N, 4) [id, vx, vy, vz]
	track_origins3d = get_velocity_origins(track_boxes3d,track_corners3d)
	for k in range(track_ids.shape[0]):
		mlab.quiver3d(track_origins3d[k, 0], track_origins3d[k, 1], track_origins3d[k, 2],
		              track_ids[k, 1], track_ids[k, 2], track_ids[k, 3],
		              scale_factor = 0.01, mode = 'arrow',
		              color=arrow_color, line_width = 2.0, figure = fig)
		mlab.text3d(track_boxes3d[k, 0], track_boxes3d[k, 1], track_boxes3d[k, 2] + track_boxes3d[k, 5] * 1.5,
		            text=str(int(track_ids[k, 0])),
		            line_width = arrow_width, color = arrow_color, figure = fig)
	return fig


def draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=1, cls=None, tag='', max_num=500, tube_radius=None):
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


def draw_grid(x1, y1, x2, y2, fig, tube_radius=0.05, color=(0.5, 0.5, 0.5)):
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


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
	# import pdb
	# pdb.set_trace()
	for x in range(bv_range[0], bv_range[2], grid_size):
		for y in range(bv_range[1], bv_range[3], grid_size):
			fig = draw_grid(x, y, x + grid_size, y + grid_size, fig, tube_radius=0.02)

	return fig


def draw_multi_grid(fig):
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
		fig = draw_grid(block_area[k, 0], block_area[k, 1], block_area[k, 3], block_area[k, 4], fig)

	# block_area = np.array([[-30, -5, 0, -10, 5, 20],
	#                        [-10, -5, 0, 10, 5, 20],
	#                        [10, -5, 0, 30, 5, 20],
	#                        [-30, -5, 20, -10, 5, 40],
	#                        [-10, -5, 20, 10, 5, 40],
	#                        [10, -5, 20, 30, 5, 40],
	#                        [-30, -5, 40, -10, 5, 60],
	#                        [-10, -5, 40, 10, 5, 60],
	#                        [10, -5, 40, 30, 5, 60],
	#                        [-30, -5, 60, -10, 5, 80],
	#                        [-10, -5, 60, 10, 5, 80],
	#                        [10, -5, 60, 30, 5, 80]], dtype=np.float32)
	# for k in range(block_area.__len__()):
	#     fig = draw_grid(block_area[k, 0], block_area[k, 2], block_area[k, 3], block_area[k, 5], fig)
	return fig


def draw_plane(fig, plane, color=(0.8, 0.5, 0)):
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

	print('Figure: ', fig)
	return fig

def draw_ground_plane(fig, ground_height = -1.60, ground_color=(0.5, 0.5, 0.5)):
	import mayavi.mlab as mlab
	x_idxs, y_idxs = np.mgrid[-70.:70.:0.1, -30.:30.:0.1]
	z_idxs = np.full_like(x_idxs, ground_height, dtype=np.float32)
	mlab.surf(x_idxs, y_idxs, z_idxs, color = ground_color, colormap='gnuplot', figure = fig)
	return fig

def draw_boxes(image, boxes, color=(255, 0, 0), scores=None):
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


def sigmoid(x):
	return 1 / (1 + math.exp(-x))


def draw_imgboxes(obj_list, img, calib, color=(0, 255, 0), use_sigmoid=False, score_thresh=0.0):
	for obj in obj_list:
		if sigmoid(obj.score) < score_thresh:
			continue

		if obj.pos.sum() == 0:
			img = draw_boxes(img, obj.box2d.reshape(-1, 4), color=color, scores=[obj.score])
		else:
			corners3d = obj.generate_corners3d()
			_, pred_boxes_corners = calib.corners3d_to_img_boxes(corners3d.reshape(1, 8, 3))
			score = sigmoid(obj.score) if use_sigmoid else obj.score
			score = '%.2f' % score
			# score = '%.2f_%.2f' % (obj.score, sigmoid(obj.score))

			cur_color = color

			img = draw_boxes(img, pred_boxes_corners.astype(np.int32), color=cur_color, scores=[score])
	return img


def draw_center_axis(fig):
	from mayavi import mlab
	def draw_each_axis(cur_pts, color, fig=fig):
		x1, y1, z1 = cur_pts[0]
		x2, y2, z2 = cur_pts[1]
		mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=color, figure=fig, tube_radius=0.08)

	# draw x axis
	pts = np.array([[0, 0, 0], [3, 0, 0]], dtype=np.float32)

	draw_each_axis(pts, color=(0, 0, 1), fig=fig)
	# draw y axis
	pts = np.array([[0, 0, 0], [0, 3, 0]], dtype=np.float32)
	draw_each_axis(pts, color=(0, 1, 0), fig=fig)
	# draw z axis
	pts = np.array([[0, 0, 0], [0, 0, 3]], dtype=np.float32)
	draw_each_axis(pts, color=(1, 0, 0), fig=fig)
	return fig


def draw_obj_axis(obj, fig, x_axis=True, y_axis=True, z_axis=True):
	from mayavi import mlab
	import lib.utils.kitti_utils as kitti_utils

	def draw_each_axis(pts, color):
		cur_pts = np.copy(pts)
		cur_pts = kitti_utils.rotate_pc_along_y(cur_pts.reshape(2, 3), -obj.ry)
		center = np.copy(obj.pos)
		center[1] -= obj.h / 2
		cur_pts += center

		x1, y1, z1 = cur_pts[0]
		x2, y2, z2 = cur_pts[1]
		mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=color, figure=fig, tube_radius=0.08)
		# mlab.quiver3d(x1, y1, z1, x2 - x1, y2 - y1, z2 - z1, figure=fig, color=color, scale_mode='vector')
		mlab.points3d(center[0], center[1], center[2], color=(1, 0.5, 0), mode='sphere', scale_factor=0.6)

	if x_axis:
		# draw x axis
		pts = np.array([[0, 0, 0], [3, 0, 0]], dtype=np.float32)
		draw_each_axis(pts, color=(0, 0, 1))
	if y_axis:
		# draw z axis
		pts = np.array([[0, 0, 0], [0, 0, 3]], dtype=np.float32)
		draw_each_axis(pts, color=(0, 1, 0))
	if z_axis:
		# draw y axis
		pts = np.array([[0, 0, 0], [0, 3, 0]], dtype=np.float32)
		draw_each_axis(pts, color=(1, 0, 0))

	return fig


def boxes3d_to_corners3d_lidar(boxes3d):
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


def create_scope_filter(pts, area_scope):
	"""
    :param pts: (N, 3) point cloud in rect camera coords
    :area_scope: (3, 2), area to keep [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
    """
	pts = pts.transpose()
	x_scope, y_scope, z_scope = area_scope[0], area_scope[1], area_scope[2]
	scope_filter = (pts[0] > x_scope[0]) & (pts[0] < x_scope[1]) \
	               & (pts[1] > y_scope[0]) & (pts[1] < y_scope[1]) \
	               & (pts[2] > z_scope[0]) & (pts[2] < z_scope[1])

	return scope_filter


def get_part_color_by_offset(pts_offset):
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


def draw_scenes(points, gt_boxes):
	fig = visualize_pts(points)
	fig = draw_multi_grid_range(fig)
	corners3d = boxes3d_to_corners3d_lidar(gt_boxes)
	fig = draw_corners3d(corners3d, fig=fig, color=(0, 1, 0))
	return fig
