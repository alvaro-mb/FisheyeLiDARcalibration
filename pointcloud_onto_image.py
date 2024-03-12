import os.path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

        translation = [0, 0, -1]
        rotation = [0, 0, 0]

        
        # Loop for selecting planes from the pointclouds and the images
        for i in indexes:
            
            # Read image and pointcloud
            image = mpimg.imread(imgs[i])
            pointcloud = Visualizer(pcls[i], imgs[i])
            
            if params.simulated:
                plt.imshow(image)
                pointcloud.define_transform_matrix(rotation, translation)
                pointcloud.get_spherical_coord()
                xs, ys, zs = pointcloud.spherical_coord[0], pointcloud.spherical_coord[1], pointcloud.spherical_coord[2]
                long = np.arctan2(ys, xs)
                lat = np.arctan2(zs, np.linalg.norm([xs, ys], axis=0))
                x = (- long) * image.shape[1] / (2*np.pi) + image.shape[1] / 2
                y = (-lat) * image.shape[0] / np.pi + image.shape[0] / 2
                plt.scatter(x, y, s=0.1, c=pointcloud.reflectivity, cmap='jet')
                plt.show()
