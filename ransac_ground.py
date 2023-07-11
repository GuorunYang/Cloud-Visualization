import os
import json
import numpy as np
import pickle as pkl
import argparse
import data_loader
import pyransac3d as pyrsc
from tqdm import tqdm

def save_pcd(cloud, save_pth):
    point_num = cloud.shape[0]
    point_type = np.dtype(
        [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
    )
    points = np.zeros(point_num, dtype = point_type)
    points['x'] = cloud[:, 0]
    points['y'] = cloud[:, 1]
    points['z'] = cloud[:, 2]
    points['intensity'] = cloud[:, 3]
    # Write the header
    with open(save_pth, 'w') as fp:
        fp.write('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1')
        fp.write('\nWIDTH ' + str(point_num))
        fp.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
        fp.write('\nPOINTS ' + str(point_num))
        fp.write('\nDATA binary')
        fp.write('\n')
    # Write the points
    with open(save_pth, 'ab+') as fp:
        pc_data = np.array(points, dtype=point_type)
        fp.write(pc_data.tostring('C'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ground estimation by RANSAC')
    parser.add_argument('-c', '--cloud', type=str, default=None, help='Cloud path')
    parser.add_argument('-t', '--thresh', type=float, default=0.2, help='Threshold distance from the plane which is considered inlier')
    parser.add_argument('--max_iteration', type=int, default=1000, help='Number of maximum iteration which RANSAC will loop over')
    parser.add_argument('--min_points', type=int, default=10000, help='Minimum points of inliers')
    parser.add_argument('--sample_num', type=int, default=-1, help='Sample frames')
    parser.add_argument('-s', '--save', type=str, default=None, help='Save pth of inlier point cloud')
    args = parser.parse_args()

    cloud_list = []
    if os.path.isdir(args.cloud):
        cloud_names = sorted(os.listdir(args.cloud))
        cloud_list = [os.path.join(args.cloud, fn) for fn in cloud_names]
    elif os.path.isfile(args.cloud):
        cloud_list.append(args.cloud)
    

    if (args.sample_num > 0) and (len(cloud_list) > args.sample_num):
        cloud_list = np.array(cloud_list)
        sample_index = np.linspace(0, len(cloud_list)-1, args.sample_num, dtype = int)
        print("sample index: ", sample_index)
        cloud_list = cloud_list[sample_index]
        cloud_list = cloud_list.tolist()

    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
        
    print("Cloud list length: ", len(cloud_list))
    inlier_heights = []
    ground_plane = pyrsc.Plane()
    for i, cloud_pth in enumerate(tqdm(cloud_list)):
        raw_cloud = data_loader.load_cloud(cloud_pth)
        cloud_points = raw_cloud[:, :3]
        # print("Cloud points shape: ", raw_cloud.shape)
        best_eq, best_inliers = ground_plane.fit(cloud_points, thresh=args.thresh, minPoints=args.min_points, maxIteration=args.max_iteration)
        # print("Best eq: ", best_eq)
        # print("Best Eq: {}x + {}y + {}z + {} = 0".format(
        #     best_eq[0], best_eq[1], best_eq[2], best_eq[3]
        # ))
        # print("All points: {}, Inliers: {}, Ratio: {:.4f}".format(
        #     raw_cloud.shape[0], best_inliers.shape[0], best_inliers.shape[0]/raw_cloud.shape[0]))
        inlier_cloud = raw_cloud[best_inliers, :]
        avg_height = np.mean(inlier_cloud[:, 2])
        inlier_heights.append(avg_height)
        if args.save is not None:
            save_pth = os.path.join(args.save, cloud_pth.split("/")[-1] + ".pcd")
            # inlier_cloud.tofile(save_pth)
            save_pcd(inlier_cloud, save_pth)

    # Compute the medium of inlier points
    median_height = np.median(inlier_heights)
    print("Inlier height: ", inlier_heights)
    print("Median height: ", median_height) 

        

