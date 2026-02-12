import numpy as np
import matplotlib.pyplot as plt
import lib.utils as utils
from lib.curvature import compute_curvature


bin_path = "/dataset/kitti-dataset/vanilla_dataset/sequences/00/velodyne/000000.bin"  # Input file
out_polar_path = "./LidarRepresentation/output/polar_image.png"  # Output image
out_cartesian_path = "./LidarRepresentation/output/cartesian_image.png"  # Output image

image_size = [128, 128]
# Load and convert
points = utils.load_pointcloud(bin_path)
points = utils.filter_z_points(points, z_range=[-1.5,1.2])

curvatures = compute_curvature(points,k_neighbors=30)
# Method 2: using np.newaxis
reshaped_curvatures = curvatures[:, np.newaxis]

# Now, you can successfully append the reshaped array
points = np.hstack((points, reshaped_curvatures))

polar_points = utils.cartesian_to_polar(points)

cartesian_image = utils.cartesian_to_image(points, img_size=image_size)
polar_image = utils.polar_to_image(polar_points, image_size)


# Save image
plt.imsave(out_cartesian_path, cartesian_image)

# Save image
plt.imsave(out_polar_path, polar_image)




