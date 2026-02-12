import numpy as np
import matplotlib.pyplot as plt
import lib.utils as utils
from lib.curvature import compute_curvature
from lib.ps_representation import cartesian_to_ps



bin_path = "/dataset/kitti-dataset/vanilla_dataset/sequences/00/velodyne/000000.bin"  # Input file
out_polar_path = "./output/ps_image2.png"  # Output image
out_cartesian_path = "./output/cartesian_image2.png"  # Output image

# Load and convert
points = utils.load_pointcloud(bin_path)
points = utils.filter_z_points(points, z_range=[-1.7,1])

curvatures = compute_curvature(points,k_neighbors=30)
# Method 2: using np.newaxis
reshaped_curvatures = curvatures[:, np.newaxis]

# Now, you can successfully append the reshaped array
points = np.hstack((points, reshaped_curvatures))

ps_points = utils.cartesian_to_spherical(points)

ps_image = utils.spherical_to_image(ps_points)
cartesian_image = utils.cartesian_to_image(points)


# Save and show image
plt.imshow(ps_image, cmap='gray')
plt.axis('off')
plt.savefig(out_polar_path, bbox_inches='tight', pad_inches=0)

# Save and show image
plt.imshow(cartesian_image, cmap='gray')
plt.axis('off')
plt.savefig(out_cartesian_path, bbox_inches='tight', pad_inches=0)



