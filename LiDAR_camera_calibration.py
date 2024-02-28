import os.path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv

from segment_anything import sam_model_registry, SamPredictor

from calibration_utils import *  # Plane, get_corners, get_rotation_and_translation, kabsch_plot, results_plot, select_image_plane, select_lidar_plane, get_lidar_corners, get_camera_corners
from pointcloud_utils import load_pc, Visualizer
from image_utils import Image
from image_utils import CamModel
from config import params


if __name__ == "__main__":
    
    # Get images and pointclouds paths
    images_path = params.images_path
    pointclouds_path = params.pointclouds_path
    imgs = sorted(glob(os.path.join(images_path, "*")), key=os.path.getmtime)
    pcls = sorted(glob(os.path.join(pointclouds_path, '*')), key=os.path.getmtime)

    # Define camera model from calibration file
    cam_model = CamModel(params.calibration_file)
    
    # Get plane parameters
    planes_sizes = params.planes_sizes
    
    # assert that images and pointclouds have the same length
    assert len(imgs) == len(pcls), "Images and pointclouds have different length"
    
    mask = None
    # Load SAM model
    if params.simulated != True:
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        sam = sam_model_registry[params.model_type](checkpoint=sam_checkpoint)
        sam.to(device=params.device)
        mask = SamPredictor(sam)
    
    # indexes = list(range(1))
    indexes = list(range(len(imgs)))

    rotations = np.zeros((len(indexes), 3))
    translations = np.zeros((len(indexes), 3))
    kabsch_errors = np.zeros(len(indexes))
    kabsch_std = np.zeros(len(indexes))
    
    pointclouds_points = [[] for _ in range(len(indexes))]
    init_planes_points = [[[] for _ in range(len(planes_sizes))] for _ in range(len(indexes))]
    selections = [[[] for _ in range(len(planes_sizes))] for _ in range(len(indexes))]
    
    # while len(indexes) != 0:
        
    # Loop for selecting planes from the pointclouds and the images
    for i in indexes:
        
        # Read image and pointcloud
        image = mpimg.imread(imgs[i])
        # Convert image to uint8 format
        image = (image * 255).astype(np.uint8)
        points = load_pc(pcls[i])
            
        # Range in meters for the lidar points
        d_range = (0, 80)

        # Define Visualizer and Image objects
        vis = Visualizer(points, image)
        if vis.reflectivity is not None:
            vis.reflectivity_filter(params.reflectivity_threshold)
        vis.get_spherical_coord(lidar2camera=0)
        vis.encode_values(d_range=d_range)
        equirect_lidar = Image(image=image, cam_model=cam_model, spherical_image=params.spherical, points_values=vis.pixel_values)
        pointclouds_points[i] = vis.lidar3d
        
        # Select planes from the pointcloud and the image
        idplane = 1
        for plane_size in planes_sizes:
            plane = Plane(plane_size[0], plane_size[1], idplane)
            
            init_plane_points = select_lidar_plane(vis, equirect_lidar, plane)
            selection_data = select_image_plane(image, plane)
            
            init_planes_points[i][idplane - 1] = init_plane_points
            selections[i][idplane - 1] = selection_data
            idplane += 1

    camera_corners2d = []
    camera_corners = []
    lidar_corners = []
    # Loop for getting corners coordinates from the pointclouds and the images
    for j in indexes:
        
        # Read image and pointcloud
        image = mpimg.imread(imgs[j])
        # Convert image to uint8 format
        image = (image[:, :, :3] * 255).astype(np.uint8)
        
        # Get corners coordinates
        idplane = 1
        for plane_size in planes_sizes:
            plane = Plane(plane_size[0], plane_size[1], idplane)
            init_plane_points = init_planes_points[j][idplane - 1]
            l_corners = get_lidar_corners(pointclouds_points[j], init_plane_points, plane)
            c_corners, c_corners2d = get_camera_corners(image, cam_model, plane, l_corners, selections[j][idplane - 1], mask)
            camera_corners.extend(c_corners)
            camera_corners2d.extend(c_corners2d)
            lidar_corners.extend(l_corners)
            idplane += 1

    camera_corners2d = np.array(camera_corners2d)  
    camera_corners = np.array(camera_corners)
    lidar_corners = np.array(lidar_corners)

    if params.save_corners:
        # concatenate camera and lidar corners
        l_c_corners = np.concatenate((lidar_corners, camera_corners2d), axis=1)
        filename = params.save_corners_path + '/' + params.save_corners_file + '.csv'
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for corner in l_c_corners:
                csvwriter.writerow(corner)

    # Get rotation and translation between camera and lidar reference systems
    rotation, translation, mean_error, std_error = get_rotation_and_translation(camera_corners, lidar_corners, PointCloud(pointclouds_points[0]))
    kabsch_errors = mean_error
    kabsch_std = std_error
    print('Mean error: ', mean_error)
    print('Std error: ', std_error)

    rotations = rotation
    translations = translation
    print('Rotation: ', rotation)
    print('Translation: ', translation)
    
    # rotation = np.zeros(3)
    # translation = np.array([0.1, -0.05, -0.16415])
            
    #     indexes = kabsch_plot(kabsch_errors, kabsch_std, label='Kabsch error: click to repeat and press enter')
    
    # # Delete the selected indexes from bad results from the Kabsch algorithm
    # delete_idx = kabsch_plot(kabsch_errors, kabsch_std, label='Kabsch error: click to delete and press enter')
    # rotations = np.delete(rotations, delete_idx, axis=0)
    # translations = np.delete(translations, delete_idx, axis=0)
    
    # Range in meters for the lidar points
    d_range = (0, 80)
    
    # Plot rotation and translation errors bars
    # rotation, translation = results_plot(rotations, translations)

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

                icorners = PointCloud(camera_corners[0 + n*i:n + n*i])
                icorners.get_spherical_coord(lidar2camera=0)
                xs, ys, zs = icorners.spherical_coord[0], icorners.spherical_coord[1], icorners.spherical_coord[2]
                long = np.arctan2(ys, xs)
                lat = np.arctan2(zs, np.linalg.norm([xs, ys], axis=0))
                x = (- long) * image.shape[1] / (2*np.pi) + image.shape[1] / 2
                y = (-lat) * image.shape[0] / np.pi + image.shape[0] / 2
                plt.scatter(x, y, s=10, c='r')
                
                pcorners = PointCloud(lidar_corners[0 + n*i:n + n*i])
                pcorners.define_transform_matrix(rotation, translation)
                pcorners.get_spherical_coord(lidar2camera=1)
                xs, ys, zs = pcorners.spherical_coord[0], pcorners.spherical_coord[1], pcorners.spherical_coord[2]
                long = np.arctan2(ys, xs)
                lat = np.arctan2(zs, np.linalg.norm([xs, ys], axis=0))
                x = (- long) * image.shape[1] / (2*np.pi) + image.shape[1] / 2
                y = (-lat) * image.shape[0] / np.pi + image.shape[0] / 2
                plt.scatter(x, y, s=10, c='g')
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

    if not os.path.exists(params.save_path):
        os.makedirs(params.save_path)
    
    if params.save_results:
        # Save rotation and translation results in a csv file
        filename = params.save_path + '/' + params.results_file + '.csv'
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # csvwriter.writerow(['Transformation matrix', pointcloud.transform_matrix])
            csvwriter.writerow(['Rotation', rotation[0], rotation[1], rotation[2]])
            csvwriter.writerow(['Translation', translation[0], translation[1], translation[2]])
            csvwriter.writerow(['MeanError', mean_error])
            csvwriter.writerow(['StdError', std_error])
        # np.savetxt(params.save_path + '/' + params.results_file, results, delimiter=",")
    