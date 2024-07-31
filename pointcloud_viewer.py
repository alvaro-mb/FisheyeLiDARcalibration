import open3d as o3d
import os
from glob import glob
from matplotlib import pyplot as plt
import numpy as np

from config import params

def save_coords(coords):
    # save coords in a csv file
    with open('coords.csv', mode='a') as file:
        file.write(','.join(coords) + '\n')

# open a folder with ply files and visualise all the poinclouds using open3d
def visualise_pointclouds(count, paths, dir):
    z = 0
    y = -4
    for path in paths:
        pcd = o3d.io.read_point_cloud(path)
        plt.scatter(np.asarray(pcd.points)[:, 1] + y, np.asarray(pcd.points)[:, 2] + z, s=5)
        z = z - 0.8
        y = y * -1
    # put the title in the window
    title = 'folder number: ' + str(count) + '\n' + dir
    # decompose directory by strings which are separated by '_'
    strings = dir.split('_')
    coords = [strings[1], strings[3], strings[5]]
    # if the plot window is clicked, print the coordinates of the point
    def onclick(event):
        save_coords(coords)
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    plt.title(title)
    # make axis equal
    plt.axis('equal')
    # make the size of the window bigger
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


if __name__ == "__main__":
    folders_path = params.pointclouds_path
    c = 0
    # loop for every directory in the folder
    for folder in os.listdir(folders_path):
        c = c + 1
        folder_path = os.path.join(folders_path, folder)
        pcls = sorted(glob(os.path.join(folder_path, '*')))
        visualise_pointclouds(c, pcls, folder)