import open3d as o3d
import os
from glob import glob

from config import params

# open a folder with ply files and visualise all the poinclouds using open3d
def visualise_pointclouds(paths):
    for path in paths:
        pcd = o3d.io.read_point_cloud(path)
        # put a title to the window
        title = os.path.basename(path)
        # make points black
        pcd.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([pcd], window_name=title)

if __name__ == "__main__":
    folder_path = params.pointclouds_path
    pcls = sorted(glob(os.path.join(folder_path, '*')))
    visualise_pointclouds(pcls)