import os
import os

# Prevent sklearn / numpy from using internal parallelism inside multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"



from multiprocessing import Pool, cpu_count

import numpy as np
import lib.utils as utils
from lib.curvature import compute_curvature
from lib.geometric_features import compute_cov_feature
from lib.other_features import compute_density, compute_albedo

POINTCLOUD_FOLDER = "/dataset/kitti-dataset/vanilla_dataset/sequences"
VELODYNE_FOLDER = "velodyne"

DATASET_FOLDER = "/dataset/kitti-dataset/channel_analysis/sequences"


#GEOMETRIC FEATURES

'''
1. Linearity = (λ1 - λ2) / λ1
2. Planarity = (λ2 - λ3) / 
3. Anisotropy = (λ1 - λ3) / λ1
4. Omnivariance = (λ1 * λ2 * λ3)^(1/3)
5. Eigenentropy = -Σ(λi * log(λi))
6. Sphericity = λ3 / λ1
'''

channels = {



    "omnivariance" : compute_cov_feature,
    "anisotropy" : compute_cov_feature, 
    "planarity" : compute_cov_feature, 
    "linearity" : compute_cov_feature, 
    "eigenentropy" : compute_cov_feature, 
    "sphericity" : compute_cov_feature, 

    #RADIOMMETRIC FEATURES

    'albedo' : compute_albedo,
    'intensity' : lambda y,x: x[:,3]/np.linalg.norm(x[:,3]),

    #STATISTICAL FEATURES
    'density' : compute_density,

    #STRUCTURAL FEATURE
    'curvature' : compute_curvature,
    'height_norm' : lambda y,x: x[:,2]/np.linalg.norm(x[:,2]),

     

}


#Select desired sequencies
sequences = ['00', '02', '06']
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
    img_size = [256,256]
    
    # Example of simplified logic inside the function:
    velodyne_pointcloud_path = os.path.join(POINTCLOUD_FOLDER, seq, VELODYNE_FOLDER)
    file_path = os.path.join(velodyne_pointcloud_path, velodyne_file)
    print(f"Processing frame {i+1}: {velodyne_file} from sequence {seq}")

    points = utils.load_pointcloud(file_path)
    points = utils.filter_z_points(points, z_range=[-1.5,1.5])
     


    #Final image where to stack channels - Starting from x and y
    points_out = points[:,0:3]
    


    for c in channels:
        f_out = channels[c](c, points)

        #f_out = f_out[:, np.newaxis]
        # Now, you can successfully append the reshaped array
        points_out[:,2] = f_out

        # Cartesian
        #-------------------------------------------------
        # Cartesian Representation - Single Channel
        cartesian_single = utils.cartesian_to_image(points[:,0:3], img_size)
        # Cartesian Representation - Multi Channel
        #cartesian_multi = utils.cartesian_to_image(points, img_size)

        utils.save_img(np.repeat(cartesian_single, 3, axis=2), i, path = os.path.join(DATASET_FOLDER, seq, c))
        #utils.save_img(cartesian_multi, i, path = os.path.join(DATASET_FOLDER, seq, 'cartesian_multi' ))
        #-------------------------------------------------

    '''

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
    range_single = utils.cartesian_to_range_image(points[:,0:3], image_size=[128,512])
    # Range Representation - Multi Channel
    range_multi = utils.cartesian_to_range_image(points, image_size=[128,512])

    utils.save_img(np.repeat(range_single, 3, axis=2), i, path = os.path.join(DATASET_FOLDER, seq, 'range_single' ))
    utils.save_img(range_multi, i, path = os.path.join(DATASET_FOLDER, seq, 'range_multi' ))
    #-------------------------------------------------
    

    
    # Front View
    #-------------------------------------------------
    # Range Representation - Single Channel
    fv_single = utils.cartesian_to_frontview(points[:,0:3], image_size=[128,256])
    # Range Representation - Multi Channel
    fv_multi = utils.cartesian_to_frontview(points, image_size=[128,256])

    utils.save_img(np.repeat(fv_single, 3, axis=2), i, path = os.path.join(DATASET_FOLDER, seq, 'fv_single'))
    utils.save_img(fv_multi, i, path = os.path.join(DATASET_FOLDER, seq, 'fv_multi'))
    #-------------------------------------------------
    
    return f"Finished frame {i+1} of sequence {seq}"
    '''

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
    num_workers = 60
    print(f"Starting processing with {num_workers} processes...")
    
    #with Pool(processes=num_workers) as pool:
        # map() blocks until all results are ready
    #    results = pool.map(process_frame, all_tasks)
        
    with Pool(processes=num_workers) as pool:
        for r in pool.imap_unordered(process_frame, all_tasks, chunksize=5):
            print(r)


    print("All processing complete.")
