import numpy as np
from sklearn.neighbors import NearestNeighbors
import lib.utils as utils

def compute_eigenvalues(point_cloud, k_neighbors=300, number_of_eigenvalues=3, use_float32=True, n_processors=15):
    """
    Optimized curvature/eigenvalue computation for 3D point clouds.
    """

    if use_float32:
        point_cloud = point_cloud.astype(np.float32)

    # --- Nearest neighbors (parallelized) ---
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree', n_jobs=1)
    nbrs.fit(point_cloud)
    distances, indices = nbrs.kneighbors(point_cloud, return_distance=True)

    neighbors = point_cloud[indices]

    # --- Center neighborhoods ---
    centroids = neighbors.mean(axis=1, keepdims=True)
    centered = neighbors - centroids

    # --- Covariance computation ---
    # More efficient einsum pattern: (N, k, 3) → (N, 3, 3)
    covariances = np.einsum('nki,nkj->nij', centered, centered) / (k_neighbors - 1)

    # --- Eigenvalues (batch processing) ---
    # Use eigh, which can handle batched inputs efficiently
    # Note: eigh assumes symmetric matrices
    
    eigenvalues = np.linalg.eigvalsh(covariances)
    #EIGENVALUES ARE SORTED IN ASCENDING ORDER !!!!!!!
    
    
    # Sort eigenvalues in descending order
    #eigenvalues = np.flip(np.sort(eigenvalues, axis=1), axis=1)

    #eigenvalues.sort(axis=1)
    return eigenvalues[:, :number_of_eigenvalues]




def compute_cov_feature(feature_str, point_cloud, k_neighbors=300):

    """
    Compute features from eigenvalues according to 
    https://isprs-archives.copernicus.org/articles/XLII-2-W15/465/2019/isprs-archives-XLII-2-W15-465-2019.pdf
    https://arxiv.org/pdf/2210.05984


    1. Linearity = (λ1 - λ2) / λ1
    2. Planarity = (λ2 - λ3) / 
    3. Anisotropy = (λ1 - λ3) / λ1
    4. Omnivariance = (λ1 * λ2 * λ3)^(1/3)
    5. Eigenentropy = -Σ(λi * log(λi))
    6. Sphericity = λ3 / λ1
    
    """
        
    eigenvalues = compute_eigenvalues(point_cloud, k_neighbors)
    lambda3, lambda2, lambda1 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]

    # Normalize eigenvalues
    sum_l = lambda1 + lambda2 + lambda3 + 1e-18
    lambda1 /= sum_l
    lambda2 /= sum_l
    lambda3 /= sum_l

    epsilon = 1e-12

    features = {
        'linearity': (lambda1 - lambda2) / (lambda1 + epsilon),
        'planarity': (lambda2 - lambda3) / (lambda1 + epsilon),
        'anisotropy': (lambda1 - lambda3) / (lambda1 + epsilon),
        'sphericity': lambda3 / (lambda1 + epsilon),
        'omnivariance': (lambda1 * lambda2 * lambda3) ** (1/3),
        'eigenentropy': -np.sum(
            np.stack([lambda1, lambda2, lambda3], axis=1) *
            np.log(np.stack([lambda1, lambda2, lambda3], axis=1) + epsilon),
            axis=1
        )
    }

    return features[feature_str]



# Example Usage
if __name__ == '__main__':
    bin_path = "/dataset/kitti-dataset/vanilla_dataset/sequences/00/velodyne/000000.bin"  # Input file
    
    out_polar_path = "./polar_image.png"  # Output image
    out_cartesian_path = "./cartesian_image.png"  # Output image

    # Load and convert
    points = utils.load_pointcloud(bin_path)
    points = utils.filter_z_points(points, z_range=[-1.5,1.2])

    number_of_eigenvalues=3
    # Compute curvature
    eigenvalues = compute_eigenvalues(points, k_neighbors=300, number_of_eigenvalues=3)
    
    print("Eigenvalues shape:", eigenvalues.shape)
    print("First 3 eigenvalues:\n", eigenvalues[:number_of_eigenvalues])    # Visualize or further process eigenvalues as needed