import os.path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv

from config import params
from pointcloud_utils import Visualizer
from image_utils import CamModel


if __name__ == "__main__":

    # Get images and pointclouds paths
    images_path = params.images_path
    pointclouds_path = params.pointclouds_path
    imgs = sorted(glob(os.path.join(images_path, "*")), key=os.path.getmtime)
    pcls = sorted(glob(os.path.join(pointclouds_path, '*')), key=os.path.getmtime)

    # Define camera model from calibration file
    cam_model = CamModel(params.calibration_file)

    filename = params.save_data_path + '/' + params.save_data_file + '.csv'
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        corners = []
        for row in reader:
            corner = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7])]
            corners.append(corner)
    corners=np.asarray(corners)
    
    lidar_corners = corners[:, :3]
    camera_corners3d = corners[:, 3:6]
    image_corners = corners[:, 6:]

    # indexes = list(range(1))
    indexes = list(range(len(imgs)))

    for i in indexes:
        
        # Read image and pointcloud
        image = mpimg.imread(imgs[i])
        n = len(params.planes_sizes) * 4
        plt.imshow(image)
        plt.scatter(image_corners[n*i:n + n*i, 0], image_corners[n*i:n + n*i, 1], s=10, c='r')
        plt.show()
        
