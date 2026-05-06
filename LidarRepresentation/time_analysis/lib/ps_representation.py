import numpy as np

def cartesian_to_ps(points):
    """Convert Nx3 cartesian points to polar coordinates (r, theta, channels)."""
    channels = points.shape[1] - 2
    x, y, z  = points[:, 0], points[:, 1], points[:,2]

    distance = np.sqrt(x**2 + y**2)
    azimuth = np.arctan2(y, x)  # angle in radians
    elevation = np.arctan2(z, x)  # angle in radians

    polar = np.stack((azimuth, elevation),axis = 1)
    
    
    polar = np.hstack((polar, distance[:, np.newaxis]))

    return polar



