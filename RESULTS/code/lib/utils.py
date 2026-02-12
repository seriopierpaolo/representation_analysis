import os
import numpy as np
import matplotlib.pyplot as plt

def load_pointcloud(bin_path, dtype=np.float32):
    """Load Nx3 point cloud from a .bin file."""
    pointcloud = np.fromfile(bin_path, dtype)
    # Try reshaping to Nx4 first
    if pointcloud.size % 4 == 0:
        pointcloud = pointcloud.reshape(-1, 4)  
        print(f"Loaded {pointcloud.shape[0]} points with X,Y,Z,I.")
        return pointcloud
    
    # If Nx4 fails, try reshaping to Nx3
    elif pointcloud.size % 3 == 0:
        pointcloud = pointcloud.reshape(-1, 3) 
        print(f"Loaded {pointcloud.shape[0]} points with X,Y,Z.")
        return pointcloud
    
    return pointcloud

import numpy as np

def load_nclt_pointcloud(bin_path):
    """
    Loads NCLT point cloud data (X, Y, Z, I) from a .bin file.
    The file is expected to contain a sequence of (X, Y, Z, I, L)
    records where X, Y, Z are 16-bit unsigned integers (H), I is 
    8-bit unsigned integer (B), and L (label) is an 8-bit unsigned
    integer (B).
    
    The function returns a numpy array with columns [X, Y, Z, I].
    """
    
    # 1. Load the raw binary data as a 1D NumPy array using the 
    #    correct combined data type for the full record (X, Y, Z, I, L).
    #    '<H' for 16-bit unsigned int (X, Y, Z), 'B' for 8-bit unsigned int (I, L)
    
    # Define the structured dtype for one record:
    # 3 x (uint16) for X, Y, Z, 2 x (uint8) for I, L
    # Total size: 2 + 2 + 2 + 1 + 1 = 8 bytes per point
    nclt_dtype = np.dtype([
        ('x_s', '<H'),
        ('y_s', '<H'),
        ('z_s', '<H'),
        ('i', '<B'),
        ('l', '<B')
    ])
    
    # Load the entire file into the structured array
    data = np.fromfile(bin_path, dtype=nclt_dtype)
    
    # Define NCLT scaling and offset constants
    scaling = 0.005 # 5 mm
    offset = -100.0

    # 2. Apply the scaling and offset to X, Y, Z
    
    # X, Y, Z are calculated using single-precision floats (np.float32)
    x = data['x_s'].astype(np.float32) * scaling + offset
    y = data['y_s'].astype(np.float32) * scaling + offset
    z = data['z_s'].astype(np.float32) * scaling + offset

    R = [[1, 0, 0],[0, -1, 0],[0, 0, -1]]
    
    # Intensity (I) is kept as a float for consistency, but the calculation is simpler
    i = data['i'].astype(np.float32)

    # 3. Combine the calculated columns (X, Y, Z, I) into a single (N, 4) array
    
    pointcloud = np.stack([x, -y, -z, i], axis=-1)
    
    return pointcloud



def filter_z_points(points, z_range = [-1, 3]):
    
    # Filter points within the specified ranges
    x,y,z,i = points[:,0], points[:,1], points[:,2], points[:,3]
    mask = (z > z_range[0]) & (z < z_range[1])
    filtered_pointcloud = np.transpose(np.stack((x[mask], y[mask], z[mask], i[mask])))

    return filtered_pointcloud


def filter_box(points, x_range = [-40,40], y_range = [-40,40], z_range = [-3, 3]):
    
    # Filter points within the specified ranges
    x,y,z,i = points[:,0], points[:,1], points[:,2], points[:,3]
    mask = (x > x_range[0]) & (x < x_range[1]) & (y > y_range[0]) & (y < y_range[1]) & (z > z_range[0]) & (z < z_range[1])
    filtered_pointcloud = np.transpose(np.stack((x[mask], y[mask], z[mask], i[mask])))

    return filtered_pointcloud


def cartesian_to_polar(points):
    """Convert Nx3 cartesian points to polar coordinates (r, theta, channels)."""
    channels = points.shape[1] - 2
    x, y = points[:, 0], points[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)  # angle in radians
    polar = np.stack((r, theta),axis = 1)
    
    for i in range(channels):
        channel_points = points[:,i+2]
        polar = np.hstack((polar, channel_points[:, np.newaxis]))

    return polar




def cartesian_to_spherical(points):
    """Convert Nx3 Cartesian points to spherical coordinates (azimuth, elevation, radius)."""
    
    channels = points.shape[1] - 2

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    r = np.sqrt(x**2 + y**2 + z**2)  # 3D range
    azimuth = np.arctan2(y, x)       # angle in XY-plane
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))  # angle from XY-plane

    spherical = np.stack((azimuth, elevation, r), axis=1)


    if channels > 0:
        extras = points[:, 3:]
        spherical = np.hstack((spherical, extras))

    return spherical






def polar_to_image(polar_points, img_size=(512, 512)):
    """Project polar coordinates with extra channels to an image."""
    

    channels = polar_points.shape[1] - 2
    img_dim = (img_size[0], img_size[1], channels)
    image = np.zeros(img_dim, dtype=np.float32)

    r, theta = polar_points[:, 0], polar_points[:, 1]

    # Normalize polar coordinates to image indices
    r_norm = (r / r.max() * (img_size[0] - 1)).astype(np.int32)
    theta_trans = theta + np.abs(theta.min())
    theta_norm = (theta_trans / theta_trans.max() * (img_size[1] - 1)).astype(np.int32)

    # Normalize each channel exactly like in cartesian_to_image
    for i in range(channels):
        value = polar_points[:, i+2]
        value_trans = value + np.abs(value.min())
        value_norm = value_trans / (value_trans.max() + 1e-10)
        image[r_norm, theta_norm, i] = value_norm

    return image





def cartesian_to_image(cartesian_points, img_size=(512, 512), information = "height"):
    """Project polar coordinates to an image."""
    
    #Keeping Z only if only two channels are provided
    if (cartesian_points.shape[1] - 2) == 2 :
        channels = 1
    else:
        channels = cartesian_points.shape[1] - 2


    
    x = - cartesian_points[:, 1]
    y = cartesian_points[:, 0]

    # Initialize image (using channel information)
    img_dim = (img_size[0], img_size[1], channels)
    image = np.zeros(img_dim, dtype=np.float32)
    
    # Normalize values to image coordinates
    x_trans = x + np.abs(x.min()) 
    y_trans = y + np.abs(y.min())

    x_norm = (x_trans / x_trans.max() * (img_size[0] - 1)).astype(np.int32)
    y_norm = (y_trans / y_trans.max() * (img_size[1] - 1)).astype(np.int32)
    
    #Add other channel information



    for i in range(channels):
        value = cartesian_points[:, i+2]
        value_trans = value + np.abs(value.min()) 
        value_norm = (value_trans / (value_trans.max()+1e-10))
        image[x_norm, y_norm, i] = value_norm
            

    return image















def cartesian_to_range_image(points, f_up=2.0, f=26.8, image_size=(64, 1024), max_range=120.0):
    """
    Converts 3D Cartesian point cloud into a normalized spherical range image
    with multiple channels, optimized for speed.

    Args:
        points (np.ndarray): NxM array, where first 3 cols are (x,y,z).
        f_up (float): Vertical FoV up in degrees.
        f (float): Vertical FoV total in degrees.
        image_size (tuple): (H,W) output image size.
        max_range (float): Maximum LiDAR range for normalization.

    Returns:
        np.ndarray: (H,W,C) float32 range image, with values in [0,1].
    """
    H, W = image_size
    num_channels = points.shape[1] - 2
    
    # Compute ranges and angles
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Compute integer pixel indices
    u = np.clip(0.5 * (1 - np.arctan2(y, x) / np.pi) * W, 0, W - 1).astype(np.int32)
    vert_angle = np.degrees(np.arcsin(z / r))
    v = np.clip((1 - (vert_angle + (f - f_up)) / f) * H, 0, H - 1).astype(np.int32)
    
    # Flattened indices for assignment
    idx = v * W + u
    
        # Other channels (1 to C)
    if num_channels > 0:
        # Initialize output image
        image = np.zeros((H * W, num_channels), dtype=np.float32)
        
        # Range channel (Channel 0)
        flat_range = np.full(H * W, max_range, dtype=np.float32)
        np.minimum.at(flat_range, idx, r)
        image[:, 0] = flat_range
        

        extras = points[:, 3:]
        # Normalize extras and assign to flattened image
        for i in range(num_channels - 1):
            channel_data = extras[:, i]
            if channel_data.max() > channel_data.min():
                image[idx, i + 1] = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
    
    # Reshape and finalize
    image = image.reshape(H, W, -1)
    
    # Normalize range and handle invalid values
    image[:, :, 0] /= max_range
    image[image[:, :, 0] == 1.0, 0] = 0.0
    
    return image


import numpy as np
import cv2


# ------------------------------------------------------------
# LiDAR vertical angle presets
# ------------------------------------------------------------

# VLP16 = LiDAR in HELILPR Dataset
# HESAIXT32 = LiDAR in Toyota Dataset
# HDL64E = LiDAR in KITTI Dataset
# HDL32E = LiDAR in NCLT Dataset


LIDAR_ELEVATION_TABLE = {
    "VLP16": np.array([ 15, 13, 11, 9, 7, 5, 3, 1,
                        -1, -3, -5, -7, -9, -11, -13, -15 ], dtype=np.float32),

    "HESAIXT32": np.array([19.433708, 18.129961, 16.821214, 15.527467, 
                           14.227470, 12.924973, 11.663726, 10.363728, 
                           9.062481, 7.759984, 6.461237, 5.162489, 
                           3.864992, 2.566245, 1.273747, -0.032500,
                           -1.339997, -2.637495, -3.937492, -5.233739, 
                           -6.529987, -7.827484, -9.126231, -10.426228, 
                           -11.719976, -12.984973, -14.274970, -15.567467, 
                           -16.859964, -18.157461, -19.451208, -20.747455], dtype=np.float32),
    
    "HDL64E": np.array([  2.        ,   1.57460317,   1.14920635,   0.72380952,
                            0.2984127 ,  -0.12698413,  -0.55238095,  -0.97777778,
                            -1.4031746 ,  -1.82857143,  -2.25396825,  -2.67936508,
                            -3.1047619 ,  -3.53015873,  -3.95555556,  -4.38095238,
                            -4.80634921,  -5.23174603,  -5.65714286,  -6.08253968,
                            -6.50793651,  -6.93333333,  -7.35873016,  -7.78412698,
                            -8.20952381,  -8.63492063,  -9.06031746,  -9.48571429,
                            -9.91111111, -10.33650794, -10.76190476, -11.18730159,
                        -11.61269841, -12.03809524, -12.46349206, -12.88888889,
                        -13.31428571, -13.73968254, -14.16507937, -14.59047619,
                        -15.01587302, -15.44126984, -15.86666667, -16.29206349,
                        -16.71746032, -17.14285714, -17.56825397, -17.99365079,
                        -18.41904762, -18.84444444, -19.26984127, -19.6952381 ,
                        -20.12063492, -20.54603175, -20.97142857, -21.3968254 ,
                        -21.82222222, -22.24761905, -22.67301587, -23.0984127 ,
                        -23.52380952, -23.94920635, -24.37460317, -24.8       ], dtype=np.float32),

    "HDL32E": np.array([ 10.67, 9.33, 8., 6.67, 5.33, 4., 2.67, 1.33, 0., -1.33,
  -2.67, -4., -5.33,  -6.66, -8., -9.33, -10.67, -12., -13.33, -14.67,
 -16., -17.33, -18.67, -20., -21.33, -22.67, -24., -25.33, -26.66, -28.,
 -29.33, -30.67], dtype=np.float32)



    
    # Add more LiDARs here:
    # "XYZ64": np.array([...])
}



"""
    "HDL32E": np.array([ 10.67, 9.33, 8., 6.67, 5.33, 4., 2.67, 1.33, 0., -1.33,
  -2.67, -4., -5.33,  -6.66, -8., -9.33, -10.67, -12., -13.33, -14.67,
 -16., -17.33, -18.67, -20., -21.33, -22.67, -24., -25.33, -26.66, -28.,
 -29.33, -30.67])
"""

""" 
"NCLTHDL32E": np.array([-0.535292, -0.511905, -0.488692, -0.465305, 
                                    -0.442092, -0.418879, -0.395666, -0.372279,
                                    -0.349066, -0.325853, -0.302466, -0.279253,
                                    -0.256040, -0.232652, -0.209440, -0.186227,
                                    -0.162839, -0.139626, -0.116239, -0.093026,
                                    -0.069813, -0.046600, -0.023213,  0.000000,
                                    0.023213, 0.046600, 0.069813, 0.093026,
                                    0.116413, 0.139626, 0.162839, 0.186227], dtype=np.float32)
"""
           
# ------------------------------------------------------------
# Cartesian → range image
# ------------------------------------------------------------
def cartesian_to_range_image_lidar_specs(points, lidar="VLP16", image_size=[64,512]):
    """
    Project 3D points into a LiDAR range image using the LiDAR's true elevation angles.
    
    Args:
        points: NxM array (x,y,z,...)
        lidar:  LiDAR model string, selects elevation table
        out_h:  Optional output height (if None, native LiDAR height is used)
        out_w:  Output width
    """




    out_h = image_size[0]
    out_w = image_size[1]
    elev = LIDAR_ELEVATION_TABLE[lidar]
 
 
    H = len(elev) if out_h is None else out_h
    W = out_w

    x, y, z = points[:,0], points[:,1], points[:,2]
    r = np.sqrt(x*x + y*y + z*z)
    max_range = r.max()
    # --------------------------
    # Horizontal angle → column
    # --------------------------
    u = (0.5 * (1 - np.arctan2(y, x) / np.pi) * W).astype(np.int32)
    u = np.clip(u, 0, W-1)

    # --------------------------
    # Vertical angle → row
    # --------------------------
    vert_angle = np.degrees(np.arcsin(z / r))
    v = np.argmin(np.abs(vert_angle[:,None] - elev[None,:]), axis=1).astype(np.int32)

    # Init image
    C = points.shape[1] - 2
    img = np.full((len(elev), W, C), 0, np.float32)

    # Flattened indexing
    idx = v * W + u

    # Range channel (minimum range per pixel)
    flat = np.full((len(elev)*W,), max_range, np.float32)
    np.minimum.at(flat, idx, r)
    img[:,:,0] = flat.reshape(len(elev), W)

    # Extra channels
    if C > 1:
        extras = points[:, 3:]
        for c in range(C-1):
            ch = img[:,:,c+1].reshape(-1)
            ch[idx] = extras[:,c]
            img[:,:,c+1] = ch.reshape(len(elev), W)

    # Normalize range
    img[:,:,0] /= max_range
    img[img[:,:,0] == 1.0, 0] = 0.0
    

    # Optional resizing
    if out_h is not None and out_h != len(elev):
        img = cv2.resize(img[:,:], (W, out_h), interpolation=cv2.INTER_LINEAR)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        

    for c in range(1, C):
        ch = img[:,:,c]
        mn, mx = ch.min(), ch.max()
        if mx > mn:
            img[:,:,c] = (ch - mn) / (mx - mn)

    return img




def cartesian_to_frontview(points,
                           lidar="VLP16",
                           image_size=(64,256),
                           fov_h=(-180.0, 180.0),
                           min_range=2.0):
    """
    Front-view projection using LiDAR's actual elevation angles.
    Returns (H, W, C) float32 with values in [0,1].
    """
    
    out_h = image_size[0]
    out_w = image_size[1]


    # -------- Vertical geometry --------
    elev = LIDAR_ELEVATION_TABLE[lidar]
    H = len(elev) if out_h is None else out_h
    W = out_w

    x, y, z = points[:,0], points[:,1], points[:,2]
    r = np.linalg.norm(points[:, :3], axis=1)
    max_range = r.max()

    # -------- Mask by range --------
    mask = (r >= min_range) & (r <= max_range)

    # -------- Compute angles --------
    az = np.degrees(np.arctan2(y, x))         # horizontal angle
    el = np.degrees(np.arcsin(z / (r+1e-10)))         # vertical angle

    # Horizontal FOV mask
    mask &= (az >= fov_h[0]) & (az <= fov_h[1])

    # Apply mask
    if not np.any(mask):
        return np.zeros((H, W, points.shape[1]-2), np.float32)

    pts, r, az, el = points[mask], r[mask], az[mask], el[mask]

    # -------- Pixel coordinates --------
    u = ( (az - fov_h[0]) / (fov_h[1] - fov_h[0]) * W ).astype(int)
    u = np.clip(u, 0, W-1)

    # Vertical: match to closest real beam angle
    v = np.argmin(np.abs(el[:,None] - elev[None,:]), axis=1)
    v = np.clip(v, 0, len(elev)-1)

    flat_idx = v * W + u

    # -------- Initialize --------
    C = pts.shape[1] - 2
    # Initialize the flattened output for all channels
    flat_img = np.zeros((len(elev) * W, C), np.float32)

    # -------- Range channel (min depth) --------
    flat_range = np.full(len(elev)*W, np.inf, np.float32)
    np.minimum.at(flat_range, flat_idx, r)

    # Normalize range and assign to flat_img (Channel 0)
    valid = np.isfinite(flat_range)
    flat_img[valid, 0] = flat_range[valid] / max_range

    # -------- Extra channels --------
    if C > 1:
        extras = pts[:, 3:]
        for ci in range(C-1):
            ch = extras[:, ci]
            # No need to normalize here, do it on the final image
            flat_img[flat_idx, ci+1] = ch

    # Reshape the flattened array into a 3D image (H_native, W, C)
    H_native = len(elev)
    img = flat_img.reshape(H_native, W, C)
    
    # -------- Final Normalization & Resizing --------
    
    # Normalize extra channels (c=1 to C-1)
    for c in range(1, C):
        ch = img[:,:,c]
        mn, mx = ch.min(), ch.max()
        # Normalize only if there is variation
        if mx > mn:
            img[:,:,c] = (ch - mn) / (mx - mn)
        # Note: Channel 0 (Range) is already normalized [0, 1]

    # Optional resizing/upsampling
    if out_h is not None and out_h != H_native:
        # cv2.resize works correctly on the 3D array (H, W, C)
        # and handles the number of channels automatically.
        img = cv2.resize(img, (W, out_h), interpolation=cv2.INTER_LINEAR)
        
        # If C=1, resize might return 2D, so check and expand back
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)


    return img



def save_img (img, i, path):
    
    os.makedirs(path, exist_ok=True)
    output_path = os.path.join(path, f"{str(i).zfill(6)}.png")
    plt.imsave(output_path, img)