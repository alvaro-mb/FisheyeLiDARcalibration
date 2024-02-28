import matplotlib.pyplot as plt
import numpy as np
import os.path
from glob import glob
import time

from image_utils import Image


if __name__ == "__main__":
    images_path = 'demo/images/'
    imgs = sorted(glob(os.path.join(images_path, "*")), key=os.path.getmtime)
    image = Image(imgs[0], fov=185, xyxy=[650, 360, 1510, 810])
    image.square_fisheye(h_cut=80)
    u_o, v_o, u_i, v_i = image.fisheye2equirect()

    # plot forward points in the image
    left = 0.35
    right = 0.45
    down = 1
    forward = 0.6
    # create an array of points in the space being x forward, y left and z down
    points_left = np.array([forward, left, -down])
    points_right = np.array([forward, -right, -down])
    array = np.arange(1, 6, 1)
    for i in array:
        points_left = np.vstack((points_left, np.array([forward+i, left, -down])))
        points_right = np.vstack((points_right, np.array([forward+i, -right, -down])))

    points = np.array([points_left, points_right])
    # get points spherical coordinates dividing by the norm
    points_norm = np.linalg.norm(points, axis=2)
    spherical_points = points / points_norm[:, :, None]

    lx, ly, colors = np.array([]), np.array([]), np.array([[], [], []]).T
    for i in range(len(spherical_points)):
        image.sphere_coord = spherical_points[i].T
        image.points_values = points_norm[i]
        lx1, ly1, colors1 = image.line_projection()
        lx = np.append(lx, lx1)
        ly = np.append(ly, ly1)
        colors = np.vstack([colors, colors1])
    for i in range(spherical_points.shape[1]):
        image.sphere_coord = spherical_points[:, i, :].T
        image.points_values = np.array([points_norm[0, i], points_norm[1, points_norm.shape[1]-1]])
        lx2, ly2, colors2 = image.line_projection()
        lx = np.append(lx, lx2)
        ly = np.append(ly, ly2)
        colors = np.vstack([colors, colors2])
    lx = lx.astype(int)
    ly = ly.astype(int)

    for img in imgs[:1]:
        start_time = time.time()
        image = Image(img, fov=185, xyxy=[650, 360, 1510, 810])
        image.square_fisheye(h_cut=80)
        image.fromfisheye2equirect(u_o, v_o, u_i, v_i)

        image.eqr_image[lx, ly, :] = colors

        if image.xyxy is not None:
            image.eqr_image = image.eqr_image[image.xyxy[1]:image.xyxy[3], image.xyxy[0]:image.xyxy[2], :]

        # print how much time it takes for the code to reach this print
        end_time = time.time()
        print('Time for reprojection:', end_time - start_time, 'seconds')

        plt.imshow(image.eqr_image)
        plt.show()
    
    # # save the equirrectangular image
    # mpi.imsave('demo/images/equirectangular_image.png', image.eqr_image)
