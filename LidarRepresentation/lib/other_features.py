import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree, BallTree
import lib.utils as utils
import lib.geometric_features


def compute_density(feature_str, point_cloud, use_float32=True, base_distance=0.3, scale_with_range=True):
    """
    Compute normalized point density using a BallTree.
    Optionally scales search radius with range (to compensate for angular sparsity).
    """
    from sklearn.neighbors import BallTree
    import numpy as np

    if use_float32:
        point_cloud = point_cloud.astype(np.float32)

    coords = point_cloud[:, :3]
    tree = BallTree(coords)

    if scale_with_range:
        ranges = np.linalg.norm(coords, axis=1)
        radii = base_distance * (ranges / np.median(ranges))  # adaptive radius per point
    else:
        radii = np.full(coords.shape[0], base_distance, dtype=np.float32)

    counts = np.zeros(coords.shape[0], dtype=np.float32)

    # Vectorized neighbor search in chunks to avoid memory blowup
    for i in range(0, len(coords), 5000):
        subset = coords[i:i+5000]
        local_radii = radii[i:i+5000]
        neighbors = tree.query_radius(subset, r=local_radii, count_only=True)
        counts[i:i+5000] = neighbors - 1  # exclude self

    # Normalize robustly
    p95 = np.percentile(counts, 95)
    density = np.clip(counts / p95, 0, 1)
    return density



def compute_albedo(feature_str, point_cloud, k_neighbors=300, use_float32=True, normalize_albedo=True):
    import numpy as np
    from sklearn.neighbors import BallTree

    if use_float32:
        point_cloud = point_cloud.astype(np.float32)

    coords = point_cloud[:, :3]
    intensities = point_cloud[:, 3]

    # Build neighbor tree
    tree = BallTree(coords)
    _, idx = tree.query(coords, k=k_neighbors)

    normals = np.zeros_like(coords, dtype=np.float32)

    # PCA normal estimation
    for i in range(coords.shape[0]):
        pts = coords[idx[i]]
        centroid = pts.mean(axis=0)
        cov = np.cov((pts - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, np.argmin(eigvals)]
        normals[i] = normal

    # Normalize normals
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # Orient normals toward sensor
    sensor_vecs = -coords
    dot_sign = np.sum(normals * sensor_vecs, axis=1)
    normals[dot_sign < 0] *= -1

    # Compute range and unit beam direction
    range_r = np.linalg.norm(coords, axis=1)
    beam_dir = sensor_vecs / range_r[:, None]

    # Compute cos(theta)
    cos_theta = np.sum(beam_dir * normals, axis=1)
    cos_theta = np.clip(cos_theta, 1e-3, 1.0)

    # Compute albedo
    albedo = intensities * (range_r ** 2) / cos_theta

    if normalize_albedo:
        p95 = np.percentile(albedo, 95)
        albedo = np.clip(albedo / p95, 0, 1)

    return albedo



#TODO: Maybe change the algotihm to compute the normals






# Example Usage
if __name__ == '__main__':
    bin_path = "/dataset/kitti-dataset/vanilla_dataset/sequences/00/velodyne/000000.bin"  # Input file
    
    out_cartesian_path = "./cartesian_image.png"  # Output image

    # Load and convert
    points = utils.load_pointcloud(bin_path)
    points = utils.filter_z_points(points, z_range=[-1.5,1.2])


    # Compute density
    #density = compute_density(points, distance=1)

    # Compute albedo
    albedo = compute_albedo(points, k_neighbors=30)
    
    print('Finish')
    



    