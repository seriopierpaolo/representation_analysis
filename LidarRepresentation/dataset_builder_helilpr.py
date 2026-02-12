import os
import struct

from multiprocessing import Pool, cpu_count

import numpy as np
import lib.utils as utils
from lib.curvature import compute_curvature

POINTCLOUD_FOLDER = "/dataset/helilpr/vanilla/"
VELODYNE_FOLDER = "Velodyne"

DATASET_FOLDER = "/dataset/helilpr/lidar_representations/representation_analysis/"


#Select desired sequencies
sequences = ['rb03']
#sequences = ['00']
num_files = 100000  # Number of frames to process (It automatically stops with the last file in the folder, so the number 100000 its only an upper limit)


def load_helilpr_pointcloud(filename):
    points = []
    with open(filename, "rb") as file:
            while True:
                # 1. Read the exact size of one point (22 bytes)
                data = file.read(22) 
                
                # 2. Check if we read a full point; if not, break the loop
                if len(data) < 22:
                    break
                    
                # 3. UNPACK THE DATA INSIDE THE LOOP
                # Unpack x, y, z, intensity (16 bytes, 'ffff')
                x, y, z, intensity = struct.unpack('ffff', data[:16])
                
                # Unpack ring (2 bytes, 'H') and time (4 bytes, 'f')
                # Note: Since you only append [x, y, z, intensity], the ring and time 
                # variables are unpacked but not used in the final list.
                # ring = struct.unpack('H', data[16:18])[0]
                # time = struct.unpack('f', data[18:22])[0]
                
                # 4. Append the desired point data
                points.append([x, y, z, intensity])
    points = np.array(points, dtype=np.float32)

    return points
    


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
    
    #img_size = [64,64]
    #img_size = [128,128]
    img_size = [256,256]
    #img_size = [512,512]
    #img_size = [1024,1024]

    
    # Example of simplified logic inside the function:
    velodyne_pointcloud_path = os.path.join(POINTCLOUD_FOLDER, seq, VELODYNE_FOLDER)
    file_path = os.path.join(velodyne_pointcloud_path, velodyne_file)
    print(f"Processing frame {i+1}: {velodyne_file} from sequence {seq}")
    base_filename, _ = os.path.splitext(velodyne_file)

    points = load_helilpr_pointcloud(file_path)
    #points = utils.filter_z_points(points, z_range=[-1.5,1.5])
    points = utils.filter_box(points, x_range=[-40, 40], y_range=[-40, 40], z_range=[-100,100])
    curvatures = compute_curvature('curvature',points,k_neighbors=50, n_processors=1) 
    reshaped_curvatures = curvatures[:, np.newaxis]
    points = np.hstack((points, reshaped_curvatures))

    
    # Cartesian
    #-------------------------------------------------
    # Cartesian Representation - Single Channel
    cartesian_single = utils.cartesian_to_image(points[:,0:3], img_size)
    # Cartesian Representation - Multi Channel
    cartesian_multi = utils.cartesian_to_image(points, img_size)

    utils.save_img(np.repeat(cartesian_single, 3, axis=2), base_filename, path = os.path.join(DATASET_FOLDER, seq, 'polar_single' ))
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
    range_single = utils.cartesian_to_range_image_lidar_specs(points[:,0:3], lidar="VLP16", image_size=[96,512])
    # Range Representation - Multi Channel
    range_multi = utils.cartesian_to_range_image_lidar_specs(points, lidar="VLP16", image_size=[96,512])

    utils.save_img(np.repeat(range_single, 3, axis=2), i, path = os.path.join(DATASET_FOLDER, seq, 'range_single' ))
    utils.save_img(range_multi, i, path = os.path.join(DATASET_FOLDER, seq, 'range_multi' ))
    #-------------------------------------------------


    
    # Front View
    #-------------------------------------------------
    # Range Representation - Single Channel
    fv_single = utils.cartesian_to_frontview(points[:,0:3], lidar="VLP16", image_size=[96,256], fov_h=(-45,45))
    # Range Representation - Multi Channel
    fv_multi = utils.cartesian_to_frontview(points, lidar="VLP16", image_size=[96,256], fov_h=(-45,45))

    utils.save_img(np.repeat(fv_single, 3, axis=2), i, path = os.path.join(DATASET_FOLDER, seq, 'fv_single'))
    utils.save_img(fv_multi, i, path = os.path.join(DATASET_FOLDER, seq, 'fv_multi'))
    #-------------------------------------------------
    
    return f"Finished frame {i+1} of sequence {seq}"


# Main execution block
if __name__ == '__main__':
    # ... define sequences, POINTCLOUD_FOLDER, etc. ...
    
    all_tasks = []
    for seq in sequences:
        velodyne_pointcloud_path = os.path.join(POINTCLOUD_FOLDER, seq, VELODYNE_FOLDER)
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
