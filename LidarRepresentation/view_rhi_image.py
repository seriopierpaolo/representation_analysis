import numpy as np
import matplotlib.pyplot as plt

def load_pointcloud(bin_path):
    """Load Nx4 point cloud from a .bin file."""
    pointcloud = np.fromfile(bin_path, dtype=np.float32)
    pointcloud = pointcloud.reshape(-1, 4)  # KITTI format: [x, y, z, intensity]
    return pointcloud

def filter_z_points(points, z_range=[-1, 3]):
    """Filter points within the specified z (height) range, preserving intensity."""
    mask = (points[:, 2] > z_range[0]) & (points[:, 2] < z_range[1])
    return points[mask]  # Keep all 4 columns including intensity

def pointcloud_to_rhi_image(pointcloud, img_size=(512, 512), save_path="rhi.png"):
    """
    Convert Nx4 point cloud (x, y, z, intensity) into a Range-Height Image (RHI).
    
    Args:
        pointcloud (np.ndarray): Nx4 array with columns [x, y, z, intensity].
        img_size (tuple): (width, height) of output image.
        save_path (str): Path to save the RHI image.
    """
    assert pointcloud.shape[1] >= 4, "Point cloud must have at least 4 columns (x, y, z, intensity)."
    
    # Extract coordinates and intensity
    x, y, z, intensity = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], pointcloud[:, 3]
    
    # Compute range (horizontal distance from origin)
    ranges = np.sqrt(x**2 + y**2)
    
    # Use z directly as height (KITTI coordinate system)
    heights = z
    
    # Define reasonable bounds for visualization
    r_min, r_max = 0, np.max(ranges)
    h_min, h_max = np.min(heights), np.max(heights)
    
    # Filter points within range bounds
    range_mask = ranges <= r_max
    ranges = ranges[range_mask]
    heights = heights[range_mask]
    intensity = intensity[range_mask]
    
    if len(ranges) == 0:
        print("Warning: No points within specified range bounds")
        return
    
    img_w, img_h = img_size
    
    # Map to image coordinates
    u = ((ranges - r_min) / (r_max - r_min) * (img_w - 1)).astype(np.int32)
    v = ((heights - h_min) / (h_max - h_min) * (img_h - 1)).astype(np.int32)  # Ground at bottom, sky at top
    v = (img_h - 1) - v  # Flip so ground is at bottom of image
    
    # Ensure coordinates are within image bounds
    valid_mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u = u[valid_mask]
    v = v[valid_mask]
    intensity = intensity[valid_mask]
    
    # Initialize image
    img = np.zeros((img_h, img_w), dtype=np.float32)
    

    
    # Normalize intensity to [0, 1]
    if len(intensity) > 0:
        intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
        
        # Handle multiple points mapping to same pixel by taking maximum intensity
        for i in range(len(u)):
            img[v[i], u[i]] =  intensity_norm[i]*255
    '''    
    # Create figure and axis with black background
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    fig.patch.set_facecolor('black')
    
    # Display image with black background for empty areas
    im = ax.imshow(img, cmap="gray", aspect='auto', vmin=0, vmax=1)
    
    # Set background color to black for NaN values
    im.set_bad(color='black')
    
    # Remove all whitespace and frames
    ax.axis('off')
    ax.set_facecolor('black')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save with no padding or borders
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, 
                facecolor='black', edgecolor='black', dpi=100)
    plt.close()
    '''
    # Save image
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0,)
    
    
    print(f"RHI image saved at {save_path}")
    print(f"Range: {r_min:.1f}m to {r_max:.1f}m")
    print(f"Height: {h_min:.1f}m to {h_max:.1f}m")
    print(f"Points plotted: {len(u)}")

    # Add after extracting coordinates
    print(f"X range: {np.min(x):.2f} to {np.max(x):.2f}")
    print(f"Y range: {np.min(y):.2f} to {np.max(y):.2f}")
    print(f"Z range: {np.min(z):.2f} to {np.max(z):.2f}")
    print(f"Range values: {np.min(ranges):.2f} to {np.max(ranges):.2f}")
    print(f"Height values: {np.min(heights):.2f} to {np.max(heights):.2f}")




# Example usage with real data
if __name__ == "__main__":
    bin_path = "/dataset/kitti-dataset/vanilla_dataset/sequences/00/velodyne/000000.bin"
    
    # Load point cloud
    pointcloud = load_pointcloud(bin_path)
    print(f"Loaded {pointcloud.shape[0]} points")
    
    # Filter points within reasonable height range
    pointcloud_filtered = filter_z_points(pointcloud, z_range=[-1.7, 1.0])
    print(f"After filtering: {pointcloud_filtered.shape[0]} points")
    
    # Generate RHI image
    pointcloud_to_rhi_image(pointcloud_filtered, img_size=(512, 512), save_path="./rhi_real.png")

