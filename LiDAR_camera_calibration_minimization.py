import os.path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv

from calibration_utils import *  # Plane, get_corners, get_rotation_and_translation, kabsch_plot, results_plot, select_image_plane, select_lidar_plane, get_lidar_corners, get_camera_corners
from pointcloud_utils import load_pc, Visualizer
from image_utils import CamModel
from minimization import get_transformation_parameters
from config import params


if __name__ == "__main__":
    
    # Get images and pointclouds paths
    images_path = params.images_path
    pointclouds_path = params.pointclouds_path
    imgs = sorted(glob(os.path.join(images_path, "*.png")))
    pcls = sorted(glob(os.path.join(pointclouds_path, '*')))

    # Define camera model from calibration file
    cam_model = CamModel(params.calibration_file)

    filename = params.save_data_path + '/' + params.save_data_file + '.csv'
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        corners = []
        for row in reader:
            corner = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])]
            corners.append(corner)
    corners=np.asarray(corners)

    lidar_corners = corners[:, :3]
    # camera_norm_corners = corners[:, 3:]
    # image = Image(camera_norm_corners, fov=185)
    # image.norm_coord = camera_norm_corners
    # image.norm2image(equirect=True)
    # camera_corners = image.eqr_coord
    camera_corners = corners[:, 3:]
    camera_corners[:, 0] = - camera_corners[:, 0]
    camera_corners[:, 1] = 2 * camera_corners[:, 1]

    if params.simulated:
        with open(params.ground_truth_file, newline='') as f:
            reader = csv.reader(f)
            # row one is rotation and row two is translation
            real_rotation = np.array([float(i) for i in next(reader)])
            real_translation = np.array([float(i) for i in next(reader)])

    # Get rotation and translation between camera and lidar reference systems
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr']
    for method in methods:
        solution, mean_error = get_transformation_parameters(lidar_corners, camera_corners, method, plot=True)
        rotation, translation = solution[:3], solution[3:]
        print('Method: ', method)
        print('Mean error: ', mean_error)
        print('Rotation: ', rotation)
        print('Translation: ', translation)
        print('')

        if not os.path.exists(params.save_path):
            os.makedirs(params.save_path)
        
        if params.save_results:
            # Save rotation and translation results in a csv file
            filename = params.save_path + '/' + params.results_file + '.csv'
            with open(filename, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                # csvwriter.writerow(['Transformation matrix', pointcloud.transform_matrix])
                csvwriter.writerow(['Method', method])
                csvwriter.writerow(['Rotation', rotation[0], rotation[1], rotation[2]])
                csvwriter.writerow(['Translation', translation[0], translation[1], translation[2]])
                if params.simulated:
                    rotationerrors = rotation - real_rotation
                    csvwriter.writerow(['RotationErrors', rotationerrors[0], rotationerrors[1], rotationerrors[2]])
                    rotationerror = np.mean(rotationerrors)
                    csvwriter.writerow(['RotationError', rotationerror])
                    translationerrors = translation - real_translation
                    csvwriter.writerow(['TranslationErrors', translationerrors[0], translationerrors[1], translationerrors[2]])
                    translationerror = np.linalg.norm(translationerrors)
                    csvwriter.writerow(['TranslationError', translationerror])
                csvwriter.writerow(['MeanPixelError', mean_error])
                # csvwriter.writerow(['StdError', std_error])
            # np.savetxt(params.save_path + '/' + params.results_file, results, delimiter=",")

    # Range in meters for the lidar points
    d_range = (0, 80)

    if params.show_lidar_onto_image > 0:
        n = 4 * len(params.planes_sizes)
        if params.simulated:
            for i in range(1):#range(len(pcls)):
                pointcloud = PointCloud(pcls[i])
                image = mpimg.imread(imgs[i])
                plt.imshow(image)
                pointcloud.define_transform_matrix(rotation, translation)
                pointcloud.get_spherical_coord()
                xs, ys, zs = pointcloud.spherical_coord[0], pointcloud.spherical_coord[1], pointcloud.spherical_coord[2]
                long = np.arctan2(ys, xs)
                lat = np.arctan2(zs, np.linalg.norm([xs, ys], axis=0))
                x = (- long) * image.shape[1] / (2*np.pi) + image.shape[1] / 2
                y = (-lat) * image.shape[0] / np.pi + image.shape[0] / 2
                plt.scatter(x, y, s=0.1, c=pointcloud.reflectivity, cmap='jet')

                # icorners = PointCloud(camera_corners[0 + n*i:n + n*i])
                # icorners.get_spherical_coord(lidar2camera=False)
                # xs, ys, zs = icorners.spherical_coord[0], icorners.spherical_coord[1], icorners.spherical_coord[2]
                # long = np.arctan2(ys, xs)
                # lat = np.arctan2(zs, np.linalg.norm([xs, ys], axis=0))
                # x = (- long) * image.shape[1] / (2*np.pi) + image.shape[1] / 2
                # y = (-lat) * image.shape[0] / np.pi + image.shape[0] / 2
                # plt.scatter(x, y, s=10, c='r')
                
                # pcorners = PointCloud(lidar_corners[0 + n*i:n + n*i])
                # pcorners.define_transform_matrix(rotation, translation)
                # pcorners.get_spherical_coord()
                # xs, ys, zs = pcorners.spherical_coord[0], pcorners.spherical_coord[1], pcorners.spherical_coord[2]
                # long = np.arctan2(ys, xs)
                # lat = np.arctan2(zs, np.linalg.norm([xs, ys], axis=0))
                # x = (- long) * image.shape[1] / (2*np.pi) + image.shape[1] / 2
                # y = (-lat) * image.shape[0] / np.pi + image.shape[0] / 2
                # plt.scatter(x, y, s=10, c='g')
                plt.show()

        else:
            for points, image, i in zip(pcls, imgs, range(len(pcls))):
                points = load_pc(points)
                image = mpimg.imread(image)
                pointcloud = Visualizer(points, image)
                pointcloud.define_transform_matrix(rotation, translation)
                pointcloud.lidar_corners = lidar_corners[0 + n*i:n + n*i]
                pointcloud.camera_corners = camera_corners[0 + n*i:n + n*i]
                pointcloud.lidar_onto_image(cam_model=cam_model, fisheye=params.show_lidar_onto_image - 1, d_range=d_range)
                plt.show()
    