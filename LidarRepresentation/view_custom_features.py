import numpy as np
import matplotlib.pyplot as plt
import lib.utils as utils
from lib.curvature import compute_curvature
from lib.geometric_features import compute_cov_feature
from lib.other_features import compute_density, compute_albedo


channels = {

    #GEOMETRIC FEATURES

    '''
    1. Linearity = (λ1 - λ2) / λ1
    2. Planarity = (λ2 - λ3) / 
    3. Anisotropy = (λ1 - λ3) / λ1
    4. Omnivariance = (λ1 * λ2 * λ3)^(1/3)
    5. Eigenentropy = -Σ(λi * log(λi))
    6. Sphericity = λ3 / λ1
    '''

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


bin_path = "/dataset/kitti-dataset/vanilla_dataset/sequences/00/velodyne/000000.bin"  # Input file

out_figure_path = "./LidarRepresentation/output/figure_image.png"  # Output image

image_size = [1024, 1024]
# Load and convert
points = utils.load_pointcloud(bin_path)
points = utils.filter_z_points(points, z_range=[-1.5,1.2])

#Final image where to stack channels - Starting from x and y
points_out = points[:,0:2]
#PROCESS DATA HERE
user_input = input("Type 3 channels separated by commas: ")
channel_names = [c.strip() for c in user_input.split(',')]
c1, c2, c3 = channel_names


for c in channel_names:
    f_out = channels[c](c, points)

    f_out = f_out[:, np.newaxis]
    # Now, you can successfully append the reshaped array
    points_out = np.hstack((points_out, f_out))


cartesian_image = utils.cartesian_to_image(points_out, img_size=image_size)

# Save image
plt.imsave(out_figure_path, cartesian_image)





# Appunti per confronto dei canali:
# Cross correlazione tra istogrammi di due canali