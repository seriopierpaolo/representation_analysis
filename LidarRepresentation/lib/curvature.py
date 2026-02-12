import numpy as np
from sklearn.neighbors import NearestNeighbors
import lib.utils as utils
from lib.geometric_features import compute_eigenvalues

''' OLD IMPLEMENTATION - SLOWER!!!
def compute_curvature(point_cloud, k_neighbors=15):
    """
    Computes the curvature of each point in a point cloud using a vectorized method.

    Args:
        point_cloud (np.array): A numpy array of shape (N, 3) representing the point cloud.
        k_neighbors (int): The number of nearest neighbors to consider for each point.

    Returns:
        np.array: A numpy array of shape (N,) containing the curvature for each point.
    """
    n_points = point_cloud.shape[0]

    # Find k-nearest neighbors for all points at once
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(point_cloud)
    distances, indices = nbrs.kneighbors(point_cloud)

    # Reshape neighbors into a (N, k, 3) tensor
    neighbors = point_cloud[indices]

    # Calculate centroids for all neighborhoods
    centroids = np.mean(neighbors, axis=1)

    # Center the neighbors for all points
    centered_neighbors = neighbors - centroids[:, np.newaxis, :]

    # Compute covariance matrices for all points at once using einsum
    # 'nij, njk -> nik' is a vectorized way to compute (X^T @ X) for each neighborhood
    # The result is an (N, 3, 3) array of covariance matrices
    covariance_matrices = np.einsum('nij, nkj -> nik', centered_neighbors, centered_neighbors) / (k_neighbors - 1)

    # Compute eigenvalues for all matrices
    # eigvalsh is faster as covariance matrices are symmetric
    eigenvalues = np.linalg.eigvalsh(covariance_matrices)

    # Sort eigenvalues in ascending order for each point
    eigenvalues.sort(axis=1)

    # Calculate curvature
    lambda_0 = eigenvalues[:, 0]
    eigen_sum = np.sum(eigenvalues, axis=1)

    # Avoid division by zero
    curvatures = np.divide(lambda_0, eigen_sum, out=np.zeros_like(lambda_0), where=eigen_sum != 0)

    curvatures_norm = (curvatures - curvatures.min()) / (curvatures.max() - curvatures.min())
    
    return curvatures_norm
'''



def compute_curvature(feature_str, point_cloud, k_neighbors=300, n_processors=15):
    """
    Computes the curvature of each point in a point cloud using a vectorized method.

    Args:
        point_cloud (np.array): A numpy array of shape (N, 3) representing the point cloud.
        k_neighbors (int): The number of nearest neighbors to consider for each point.

    Returns:
        np.array: A numpy array of shape (N,) containing the curvature for each point.
    """

    eigenvalues = compute_eigenvalues(point_cloud, k_neighbors=k_neighbors, number_of_eigenvalues=3, n_processors= n_processors)


    # Sort eigenvalues in ascending order for each point
    eigenvalues.sort(axis=1)

    # Calculate curvature
    lambda_0 = eigenvalues.min(axis=1)
    eigen_sum = np.sum(eigenvalues, axis=1)

    # Avoid division by zero
    curvatures = np.divide(lambda_0, eigen_sum, out=np.zeros_like(lambda_0), where=eigen_sum != 0)

    curvatures_norm = (curvatures - curvatures.min()) / (curvatures.max() - curvatures.min()+1e-10)
    
    return curvatures_norm




# Example Usage
if __name__ == '__main__':
    bin_path = "/dataset/kitti-dataset/vanilla_dataset/sequences/00/velodyne/000000.bin"  # Input file
    out_polar_path = "./polar_image.png"  # Output image
    out_cartesian_path = "./cartesian_image.png"  # Output image

    # Load and convert
    points = utils.load_pointcloud(bin_path)
    points = utils.filter_z_points(points, z_range=[-1.5,1.2])

    # Compute curvature
    curvatures = compute_curvature(points, k_neighbors=300)
    print("Computed Curvature for each point:", curvatures)
    
    # The curvature values should be close to zero for a sphere, as it is a smooth surface.
    # High curvature values would be found at corners or sharp edges.
    print(f"\nAverage curvature: {np.mean(curvatures):.4f}")
    print(f"Standard deviation of curvature: {np.std(curvatures):.4f}")

    # Method 2: using np.newaxis
    reshaped_curvatures = curvatures[:, np.newaxis]

    # Now, you can successfully append the reshaped array
    result = np.hstack((points, reshaped_curvatures))

    cartesian_image = utils.cartesian_to_image(result)

    import matplotlib.pyplot as plt
    out_cartesian_path = "./output/curvature.png"  # Output image
    # Save and show image
    plt.imshow(cartesian_image, cmap='gray')
    plt.axis('off')
    plt.savefig(out_cartesian_path, bbox_inches='tight', pad_inches=0)
