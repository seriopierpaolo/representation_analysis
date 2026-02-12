import os
import struct

from multiprocessing import Pool, cpu_count

import numpy as np
import lib.utils as utils
from lib.curvature import compute_curvature

POINTCLOUD_FOLDER = "/dataset/toyota-dataset/vanilla/"
PC_FOLDER = "pcs"

DATASET_FOLDER = "/dataset/toyota-dataset/lidar_representations/representation_analysis/"

sequences = ['00']

num_files = 100000  # Number of frames to process (It automatically stops with the last file in the folder, so the number 100000 its only an upper limit)



def load_toyota_pointcloud(bin_path: str) -> np.ndarray:
    """
    Reads a point cloud file (.bin) formatted as XYZI (4 channels of float32) 
    in the Point-Major (Interleaved) format.

    Args:
        bin_path: The full path to the .bin file (e.g., 'pc/000001.bin').

    Returns:
        A NumPy array (N x 4) where N is the number of points and the columns
        are [X, Y, Z, I] before transformation. Returns an empty array if the 
        file is not found.
    """
    if not os.path.exists(bin_path):
        print(f"Error: File not found at {bin_path}")
        return np.array([])

    # --- CRITICAL FIX: Assume Point-Major (Interleaved) Format ---
    # Based on the successful alternative load functions provided by the user, 
    # the data is stored as X1, Y1, Z1, I1, X2, Y2, Z2, I2, ...
    
    # Read the raw binary data and directly reshape to (N x 4)
    data = np.fromfile(bin_path, dtype=np.float32)
    
    # We must check if the total number of elements is divisible by 4.
    if data.size % 4 != 0:
        print(f"Error: File size {data.size} is not divisible by 4 (XYZI). Check file integrity.")
        return np.array([])
        
    # The resulting point_cloud is now correctly structured as [X, Y, Z, I]
    point_cloud = data.reshape(-1, 4)
    # -----------------------------------------------------------
    
    
    # --- BEV TRANSFORMATION (REQUIRED FOR BLACK-BOX C2I) ---
    # The non-editable 'cartesian_to_image' function (C2I) maps:
    # Row Index (Vertical) <- -Input[1]
    # Col Index (Horizontal) <- Input[0]
    
    # Goal: Input structure must be [Y, -X, Z, I] 
    # This ensures: 
    # 1. Row Index <- -(-X) = X (Forward)
    # 2. Col Index <- Y (Lateral)
    
    # Step 1: Swap X (col 0) and Y (col 1). Array structure is now [Y, X, Z, I].
    #point_cloud[:, [0, 1]] = point_cloud[:, [1, 0]]
    
    # Step 2: Invert the new X (Original X) at col 1.
    # Array structure is now [Y, -X, Z, I].
    #point_cloud[:, 1] = -point_cloud[:, 1]
    
    # -----------------------------------------------------------------------
    
    return point_cloud
    

    


'''
for seq in sequences:
    velodyne_pointcloud_path = os.path.join(POINTCLOUD_FOLDER, seq, VELODYNE_FOLDER)

    # Get sorted list of velodyne files
    velodyne_files = sorted([f for f in os.listdir(velodyne_pointcloud_path) if f.endswith('.bin')])[:num_files]

    # Loop through the files and process them
    for i, velodyne_file in enumerate(velodyne_files):
        print(f"Processing frame {i+1}/{num_files}: {velodyne_file}")
        file_path = os.path.join(velodyne_pointcloud_path, velodyne_file)

        # Load and convert
        points = utils.load_pointcloud(file_path)
        points = utils.filter_z_points(points, z_range=[-1.5,1.2])

        curvatures = compute_curvature(points,k_neighbors=30)
        
        reshaped_curvatures = curvatures[:, np.newaxis]

        points = np.hstack((points, reshaped_curvatures))

        


        # Cartesian
        #-------------------------------------------------
        # Cartesian Representation - Single Channel
        cartesian_single = utils.cartesian_to_image(points[:,0:3])
        # Cartesian Representation - Multi Channel
        cartesian_multi = utils.cartesian_to_image(points)

        utils.save_img(cartesian_single, i, path = os.path.join(DATASET_FOLDER, seq, 'cartesian_single' ))
        utils.save_img(cartesian_multi, i, path = os.path.join(DATASET_FOLDER, seq, 'cartesian_multi' ))
        #-------------------------------------------------

        # Polar
        #-------------------------------------------------
        polar_points = utils.cartesian_to_polar(points)
        # Polar Representation - Single Channel
        polar_single = utils.polar_to_image(polar_points[:,0:3])
        # Polar Representation - Multi Channel
        polar_multi = utils.polar_to_image(polar_points)

        utils.save_img(polar_single, i, path = os.path.join(DATASET_FOLDER, seq, 'polar_single' ))
        utils.save_img(polar_multi, i, path = os.path.join(DATASET_FOLDER, seq, 'polar_multi' ))
        #-------------------------------------------------

        # Range
        #-------------------------------------------------
        # Range Representation - Single Channel
        range_single = utils.cartesian_to_range_image(points[:,0:3])
        # Range Representation - Multi Channel
        range_multi = utils.cartesian_to_range_image(points)

        utils.save_img(range_single, i, path = os.path.join(DATASET_FOLDER, seq, 'range_single' ))
        utils.save_img(range_multi, i, path = os.path.join(DATASET_FOLDER, seq, 'range_multi' ))
        #-------------------------------------------------


        # Front View
        #-------------------------------------------------
        # Range Representation - Single Channel
        fv_single = utils.cartesian_to_frontview(points[:,0:3])
        # Range Representation - Multi Channel
        fv_multi = utils.cartesian_to_frontview(points)

        utils.save_img(fv_single, i, path = os.path.join(DATASET_FOLDER, seq, 'fv_single'))
        utils.save_img(fv_multi, i, path = os.path.join(DATASET_FOLDER, seq, 'fv_multi'))
        #-------------------------------------------------
'''

def process_frame(args):
    # Unpack arguments
    i, file, seq = args
    
    #img_size = [64,64]
    #img_size = [128,128]
    img_size = [256,256]
    #img_size = [512,512]
    #img_size = [1024,1024]

    
    # Example of simplified logic inside the function:
    pc_pointcloud_path = os.path.join(POINTCLOUD_FOLDER, seq, PC_FOLDER)
    file_path = os.path.join(pc_pointcloud_path, file)
    print(f"Processing frame {i+1}: {file} from sequence {seq}")
    base_filename, _ = os.path.splitext(file)

    points = load_toyota_pointcloud(file_path)
    #points = utils.filter_z_points(points, z_range=[-1.5,1.5])
    points = utils.filter_box(points, x_range=[-40,40], y_range=[-40,40], z_range=[-1.5,5])
    curvatures = compute_curvature('curvature', points,k_neighbors=30, n_processors=1) 
    reshaped_curvatures = curvatures[:, np.newaxis]
    points = np.hstack((points, reshaped_curvatures))

    
    # Cartesian
    #-------------------------------------------------
    # Cartesian Representation - Single Channel
    cartesian_single = utils.cartesian_to_image(points[:,0:3], img_size)
    # Cartesian Representation - Multi Channel
    cartesian_multi = utils.cartesian_to_image(points, img_size)

    utils.save_img(np.repeat(cartesian_single, 3, axis=2), base_filename, path = os.path.join(DATASET_FOLDER, seq, 'cartesian_single'))
    utils.save_img(cartesian_multi, i, path = os.path.join(DATASET_FOLDER, seq, 'cartesian_multi' ))
    #-------------------------------------------------


    

    # Polar
    #-------------------------------------------------
    polar_points = utils.cartesian_to_polar(points)
    # Polar Representation - Single Channel
    polar_single = utils.polar_to_image(polar_points[:,0:3], img_size)
    # Polar Representation - Multi Channel
    polar_multi = utils.polar_to_image(polar_points, img_size)

    utils.save_img(np.repeat(polar_single, 3, axis=2), i, path = os.path.join(DATASET_FOLDER, seq, 'polar_single' ))
    utils.save_img(polar_multi, i, path = os.path.join(DATASET_FOLDER, seq, 'polar_multi' ))
    #-------------------------------------------------
    

    # Range
    #-------------------------------------------------
    # Range Representation - Single Channel
    range_single = utils.cartesian_to_range_image_lidar_specs(points[:,0:3], lidar="HESAIXT32", image_size=[96,512])
    # Range Representation - Multi Channel
    range_multi = utils.cartesian_to_range_image_lidar_specs(points, lidar="HESAIXT32", image_size=[96,512])

    utils.save_img(np.repeat(range_single, 3, axis=2), i, path = os.path.join(DATASET_FOLDER, seq, 'range_single' ))
    utils.save_img(range_multi, i, path = os.path.join(DATASET_FOLDER, seq, 'range_multi' ))
    #-------------------------------------------------


    
    # Front View
    #-------------------------------------------------
    # Range Representation - Single Channel
    fv_single = utils.cartesian_to_frontview(points[:,0:3], lidar="HESAIXT32", image_size=[96,256])
    # Range Representation - Multi Channel
    fv_multi = utils.cartesian_to_frontview(points, lidar="HESAIXT32", image_size=[96,256])

    utils.save_img(np.repeat(fv_single, 3, axis=2), i, path = os.path.join(DATASET_FOLDER, seq, 'fv_single'))
    utils.save_img(fv_multi, i, path = os.path.join(DATASET_FOLDER, seq, 'fv_multi'))
    #-------------------------------------------------
    
    return f"Finished frame {i+1} of sequence {seq}"


# Main execution block
if __name__ == '__main__':
    # ... define sequences, POINTCLOUD_FOLDER, etc. ...
    
    all_tasks = []
    for seq in sequences:
        velodyne_pointcloud_path = os.path.join(POINTCLOUD_FOLDER, seq, PC_FOLDER)
        velodyne_files = sorted([f for f in os.listdir(velodyne_pointcloud_path) if f.endswith('.bin')])[:num_files]
        
        # Collect all tasks (index, file name, sequence)
        for i, velodyne_file in enumerate(velodyne_files):
            all_tasks.append((i, velodyne_file, seq))

    # Use a process pool with a suitable number of workers (e.g., all available cores)
    num_workers = 50
    print(f"Starting processing with {num_workers} processes...")
    
    #with Pool(processes=num_workers) as pool:
        # map() blocks until all results are ready
    #    results = pool.map(process_frame, all_tasks)
        
    with Pool(processes=num_workers) as pool:
        for r in pool.imap_unordered(process_frame, all_tasks, chunksize=5):
            print(r)


    print("All processing complete.")
