import os
import numpy as np
import matplotlib.pyplot as plt


def load_pointcloud(bin_path):
    """Load Nx3 point cloud from a .bin file."""
    pointcloud = np.fromfile(bin_path, dtype=np.float32)
    pointcloud = pointcloud.reshape(-1, 4)  # Ensure Nx3
    return pointcloud






def filter_z_points(points, z_range = [-1, 3]):
    
    # Filter points within the specified ranges
    x,y,z,i = points[:,0], points[:,1], points[:,2], points[:,3]
    mask = (z > z_range[0]) & (z < z_range[1])
    filtered_pointcloud = np.transpose(np.stack((x[mask], y[mask], z[mask], i[mask])))

    return filtered_pointcloud
