import numpy as np
import matplotlib.pyplot as plt
import lib.utils as utils
from lib.curvature import compute_curvature



bin_path = "/dataset/kitti-dataset/vanilla_dataset/sequences/00/velodyne/000000.bin"  # Input file
out_range_image_path = "./LidarRepresentation/output/range_image.png"  # Output image
out_front_view_path = "./LidarRepresentation/output/front_view.png"  # Output image

# Load and convert
points = utils.load_pointcloud(bin_path)
points = utils.filter_z_points(points, z_range=[-1.5,1.2])

curvatures = compute_curvature(points,k_neighbors=30)
# Method 2: using np.newaxis
reshaped_curvatures = curvatures[:, np.newaxis]

# Now, you can successfully append the reshaped array
points = np.hstack((points, reshaped_curvatures))

#ps_points = utils.cartesian_to_spherical(points)

range_image = utils.cartesian_to_range_image(points,image_size=[128,512])
#range_image = utils.projection_to_image(range_image, img_size=(64,1024))
#cartesian_image = utils.cartesian_to_image(points)


# Save and show image
plt.imsave(out_range_image_path, range_image)


# Save and show image
#plt.imshow(cartesian_image, cmap='gray')
#plt.axis('off')
#plt.savefig(out_cartesian_path, bbox_inches='tight', pad_inches=0)



front_view = utils.cartesian_to_frontview(points, image_size=(256,512))
# Save and show image
plt.imsave(out_front_view_path, front_view)
