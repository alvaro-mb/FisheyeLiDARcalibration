import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement

from robots.simulation import Simulation
from robots.camera import Camera
from robots.velodyne import Velodyne
from robots.objects import CoppeliaObject


def capture_image(i):
    camera = Camera(simulation=simulation)
    camera.start(name='/sphericalVisionRGB/sensor')

    image = camera.get_image()
    # make the index of the image name always 4 digits
    index = str(i).zfill(4)
    camera.save_image(f'/home/arvc/Alvaro/calibration_simulation/Appications/pyARTE/simulation/images/image{index}.png')

def capture_lidar(i):
    lidar = Velodyne(simulation=simulation)
    lidar.start(name='/VelodyneVPL16')

    data = lidar.get_laser_data()
    print('Received Laser Data')
    try:
        print(data.shape)
        print(data.dtype)
    except:
        print('Unknown data type')

    # Define the point cloud object with open3d
    pointcloud = o3d.geometry.PointCloud()
    # Convert the data to a numpy array
    pcd_array = np.asarray(data)
    # Convert the numpy array to a point cloud and visualize the point cloud
    # pointcloud.points = o3d.utility.Vector3dVector(pcd_array)
    # o3d.visualization.draw_geometries([pointcloud])
    # Save the point cloud as a .ply file
    index = str(i).zfill(4)
    plycloud = np.array(list(map(tuple, pcd_array)), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    plycloud = PlyElement.describe(plycloud, 'vertex')
    PlyData([plycloud]).write(f'/home/arvc/Alvaro/calibration_simulation/Appications/pyARTE/simulation/pointclouds/pointcloud{index}.ply')
    # o3d.io.write_point_cloud(f'/home/arvc/Alvaro/calibration_simulation/Appications/pyARTE/simulation/pointclouds/pointcloud{index}.ply', pointcloud)

def capture(i):
    capture_lidar(i)
    capture_image(i)

def move_objects():
    bpx = 4 + np.random.uniform(-0.5, 0.5)
    bpy = 1.5 + np.random.uniform(-0.5, 1)
    bpz = 1 + np.random.uniform(0, 0.2)
    spx = 1.35 + np.random.uniform(-0.75, 0.75)
    spy = -1.35 + np.random.uniform(-0.75, 0.75)
    spz = 0.75 + np.random.uniform(-0.2, 0.2)
    rot1 = np.pi/2
    rot2 = np.random.uniform(np.pi/3, 3*np.pi/4)
    rot3 = np.random.uniform(-np.pi/18, np.pi/18)
    rot4 = np.pi/2
    rot5 = np.random.uniform(np.pi/10, 4*np.pi/10)
    rot6 = np.random.uniform(-np.pi, np.pi)
    big_plane.set_position([0, 0, 0])
    big_plane.set_orientation([rot1, rot2, rot3])
    big_plane.set_position([bpx, bpy, bpz])
    small_plane.set_position([0, 0, 0])
    small_plane.set_orientation([rot4, rot5, rot6])
    small_plane.set_position([spx, spy, spz])


if __name__ == "__main__":
    # start simulation
    simulation = Simulation()
    simulation.start()
    simulation.wait()

    global big_plane, small_plane

    big_plane = CoppeliaObject(simulation=simulation)
    big_plane.start(name='/Plane[0]')
    small_plane = CoppeliaObject(simulation=simulation)
    small_plane.start(name='/Plane[1]')

    for i in range(20):
        capture(i)
        print('Capture done!')
        move_objects()

    simulation.stop()
    print('Simulation stopped')
    exit(0)