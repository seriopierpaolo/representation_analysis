import os

from multiprocessing import Pool, cpu_count

import numpy as np
import lib.utils as utils
from lib.curvature import compute_curvature

import pandas as pd

POINTCLOUD_FOLDER = "/dataset/nclt-dataset/vanilla/pointcloud/"
VELODYNE_FOLDER = "velodyne_sync"

DATASET_FOLDER = "/dataset/nclt-dataset/lidar_representations/representation_analysis/"
GROUNDTRUTH_FOLDER = "/dataset/nclt-dataset/vanilla/poses/"

#Select desired sequencies
sequences = ['2012-01-15', '2012-02-04', '2012-03-17', '2012-06-15', '2012-09-28', '2012-11-16', '2013-02-23']
#sequences = ['2012-01-15']

num_files = 100000  # Number of frames to process (It automatically stops with the last file in the folder, so the number 100000 its only an upper limit)

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
    i, velodyne_file, seq = args
    
    img_size = [64,64]
    img_size = [128,128]
    img_size = [256,256]
    #img_size = [512,512]
    #img_size = [1024,1024]

    
    # Example of simplified logic inside the function:
    velodyne_pointcloud_path = os.path.join(POINTCLOUD_FOLDER, seq, VELODYNE_FOLDER)
    file_path = os.path.join(velodyne_pointcloud_path, velodyne_file)
    print(f"Processing frame {i+1}: {velodyne_file} from sequence {seq}")

    points = utils.load_nclt_pointcloud(file_path)
    #points = utils.filter_z_points(points, z_range=[-1.5,1.5])
    points = utils.filter_box(points, x_range=[-40,40], y_range=[-40,40], z_range=[-15,15])
    curvatures = compute_curvature('curvature', points, k_neighbors=20, n_processors=1) 
    reshaped_curvatures = curvatures[:, np.newaxis]
    points = np.hstack((points, reshaped_curvatures))

    
    # Cartesian
    #-------------------------------------------------
    # Cartesian Representation - Single Channel
    cartesian_single = utils.cartesian_to_image(points[:,0:3], img_size)
    # Cartesian Representation - Multi Channel
    cartesian_multi = utils.cartesian_to_image(points, img_size)

    utils.save_img(np.repeat(cartesian_single, 3, axis=2), i, path = os.path.join(DATASET_FOLDER, seq, 'cartesian_single' ))
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
    range_single = utils.cartesian_to_range_image_lidar_specs(points[:,0:3], lidar='HDL32E', image_size=[96,512])
    # Range Representation - Multi Channel
    range_multi = utils.cartesian_to_range_image_lidar_specs(points, lidar='HDL32E', image_size=[96,512])

    utils.save_img(np.repeat(range_single, 3, axis=2), i, path = os.path.join(DATASET_FOLDER, seq, 'range_single' ))
    utils.save_img(range_multi, i, path = os.path.join(DATASET_FOLDER, seq, 'range_multi' ))
    #-------------------------------------------------


    
    # Front View
    #-------------------------------------------------
    # Range Representation - Single Channel
    fv_single = utils.cartesian_to_frontview(points[:,0:3], lidar='HDL32E', image_size=[96,256], fov_h=(-45,45))
    # Range Representation - Multi Channel
    fv_multi = utils.cartesian_to_frontview(points, lidar='HDL32E', image_size=[96,256], fov_h=(-45,45))

    utils.save_img(np.repeat(fv_single, 3, axis=2), i, path = os.path.join(DATASET_FOLDER, seq, 'fv_single'))
    utils.save_img(fv_multi, i, path = os.path.join(DATASET_FOLDER, seq, 'fv_multi'))
    #-------------------------------------------------
    
    return f"Finished frame {i+1} of sequence {seq}"


# Main execution block
if __name__ == '__main__':

    
    all_tasks = []
    

    all_tasks = []

    for seq in sequences:
        # Path to ground truth text file for this sequence
        gt_txt_path = os.path.join(GROUNDTRUTH_FOLDER, seq + '.txt')

        # Load text file (assuming space- or tab-separated columns)
        gt_data = pd.read_csv(gt_txt_path, delim_whitespace=True, header=None)

        # The first column should be the timestamp (float-like)
        for i, row in gt_data.iterrows():
            # Convert from float-like string "1.36164918884706" -> "136164918884706"
            raw_timestamp = str(row[0])  # first column
            timestamp = raw_timestamp.replace('.', '')[:16]  # trim/pad to 16 digits if needed

            # Construct expected .bin filename
            velodyne_pointcloud_path = os.path.join(POINTCLOUD_FOLDER, seq, VELODYNE_FOLDER)
            velodyne_file = f"{timestamp}.bin"
            velodyne_path = os.path.join(velodyne_pointcloud_path, velodyne_file)

            if not os.path.isfile(velodyne_path):
                print(f"Warning: Missing pointcloud file {velodyne_file} in {seq}")
                continue

            # Add (index, filename, sequence) to tasks
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
