import os.path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv

from config import params
from pointcloud_utils import Visualizer


if __name__ == "__main__":
        
        # Get images and pointclouds paths
        images_path = params.images_path
        pointclouds_path = params.pointclouds_path
        imgs = sorted(glob(os.path.join(images_path, "*")))
        pcls = sorted(glob(os.path.join(pointclouds_path, '*')))
    
        # indexes = list(range(1))
        indexes = list(range(len(imgs)))

        rotation = [0, 0, 0]
        translation = [0, 0, 0]

        # if params.simulated:
        #     with open(params.ground_truth_file, newline='') as f:
        #         reader = csv.reader(f)
        #         # row one is rotation and row two is translation
        #         rotation = np.array([float(i) for i in next(reader)])
        #         translation = np.array([float(i) for i in next(reader)])
        
        # Loop for selecting planes from the pointclouds and the images
        for i in indexes:
            
            # Read image and pointcloud
            image = mpimg.imread(imgs[i])
            pointcloud = Visualizer(pcls[i], imgs[i])
            
            if params.simulated:
                plt.imshow(image)
                # pointcloud.define_transform_matrix(rotation, translation)
                pointcloud.get_spherical_coord(False)
                xs, ys, zs = pointcloud.spherical_coord[0], pointcloud.spherical_coord[1], pointcloud.spherical_coord[2]
                long = np.arctan2(ys, xs)
                lat = np.arctan2(zs, np.linalg.norm([xs, ys], axis=0))
                x = (- long) * image.shape[1] / (2*np.pi) + image.shape[1] / 2
                y = (-lat) * image.shape[0] / np.pi + image.shape[0] / 2
                plt.scatter(x, y, s=1.3, c=pointcloud.reflectivity, cmap='jet')
                plt.show()
