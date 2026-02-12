import numpy as np
import matplotlib.pyplot as plt
import os

import lib.utils as utils
from lib.curvature import compute_curvature
from lib.geometric_features import compute_cov_feature
from lib.other_features import compute_density, compute_albedo

# --- Feature Channel Definitions ---
channels = {
    "omnivariance" : compute_cov_feature, "anisotropy" : compute_cov_feature, 
    "planarity" : compute_cov_feature, "linearity" : compute_cov_feature, 
    "eigenentropy" : compute_cov_feature, "sphericity" : compute_cov_feature, 
    'albedo' : compute_albedo,
    'intensity' : lambda y,x: x[:,3]/np.linalg.norm(x[:,3]),
    'density' : compute_density, 
    'curvature' : compute_curvature,
    'height_norm' : lambda y,x: x[:,2]/np.linalg.norm(x[:,2]),
}

# --- Configuration ---
bin_path = "/dataset/kitti-dataset/vanilla_dataset/sequences/00/velodyne/000000.bin"
out_figure_dir = "./"
image_size = [256, 256]
os.makedirs(out_figure_dir, exist_ok=True)

# --- Data Loading (with z-filtering) ---
points = utils.load_pointcloud(bin_path)
points = utils.filter_z_points(points, z_range=[-1.5, 1.2])

# Example input
user_input = "height_norm,intensity,curvature"
channel_names = [c.strip() for c in user_input.split(',')]

# --- Core Processing ---
feature_maps_2d = []
all_data_mask = np.zeros(image_size, dtype=bool)

for c_name in channel_names:
    f_out = channels[c_name](c_name, points)
    f_min, f_max = np.min(f_out), np.max(f_out)
    f_out_norm = (f_out - f_min) / (f_max - f_min) if f_max != f_min else np.zeros_like(f_out)

    points_out = np.hstack((points, f_out_norm[:, np.newaxis]))
    feature_image_2d = utils.cartesian_to_image(points_out, img_size=image_size)

    feature_map = feature_image_2d[:, :, 2]
    feature_maps_2d.append(feature_map)

    all_data_mask = np.logical_or(all_data_mask, feature_map > 0.0)

# === Colors ===
sky_blue = np.array([0.53, 0.81, 0.92])  # brighter visible blue
rgb_letters = ["R", "G", "B"]

# ===========================
#   SAVE 3 INDEPENDENT IMAGES
# ===========================
for idx, feature_map in enumerate(feature_maps_2d):

    rgb_image = np.zeros((*image_size, 3), dtype=float)

    if idx == 2:
        # Use sky blue instead of pure blue for visibility
        rgb_image = feature_map[..., None] * sky_blue
    else:
        # Standard R or G channel
        rgb_image[:, :, idx] = feature_map

    out_path = os.path.join(out_figure_dir, f"{channel_names[idx]}_{rgb_letters[idx]}.png")
    plt.imsave(out_path, rgb_image)
    print(f"Saved single-channel {rgb_letters[idx]} image: {out_path}")


# ===========================
#   SAVE COMBINED RGB IMAGE
# ===========================
combined_rgb = np.zeros((*image_size, 3), dtype=float)

# R channel
combined_rgb[:, :, 0] = feature_maps_2d[0]

# G channel
combined_rgb[:, :, 1] = feature_maps_2d[1]

# B channel → using sky blue scaling
combined_rgb[:, :, 2] = feature_maps_2d[2]

# — Mask background to keep it black —
combined_rgb[~all_data_mask] = 0.0

combined_out = os.path.join(out_figure_dir,
                            f"{channel_names[0]}_{channel_names[1]}_{channel_names[2]}_combined.png")
plt.imsave(combined_out, combined_rgb)

print(f"Saved combined RGB stacked image: {combined_out}")
