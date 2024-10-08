import numpy as np
from numpy import ndarray
from scipy.spatial import KDTree


def icp(source_points: ndarray, target_points: ndarray, max_iterations=100, tolerance=1e-6):
    """
    ICP algorithm for 2D point clouds to find the optimal rotation and translation between them.
    :param source_points: Nx2 array of source points
    :param target_points: Nx2 array of target points
    :param max_iterations: Maximum number of iterations
    :param tolerance: Tolerance for convergence
    :return: Returns the rotation in radians and translation vector
    """
    # Initial transformation with identity matrix and zero vector
    rotation_matrix = np.eye(2)
    translation_vector = np.zeros((2,))

    for _ in range(max_iterations):
        # Apply the current transformation to source points
        transformed_points = np.dot(source_points, rotation_matrix.T) + translation_vector

        # Find nearest neighbors in target point cloud
        tree = KDTree(target_points)
        _, indices = tree.query(transformed_points)

        # Corresponding points from target
        matched_points = target_points[indices]

        # Compute centroids of both point sets
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(matched_points, axis=0)

        # Center the points around the centroids
        source_centered = source_points - source_centroid
        target_centered = matched_points - target_centroid

        # Compute covariance matrix
        cov = np.dot(source_centered.T, target_centered)

        # Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(cov)

        # Compute optimal rotation
        new_rotation_matrix = np.dot(U, Vt)

        # Ensure it's a proper rotation (det(rotation_matrix) = 1)
        if np.linalg.det(new_rotation_matrix) < 0:
            Vt[1, :] *= -1
            new_rotation_matrix = np.dot(U, Vt)

        # Compute optimal translation
        new_translation_vector = target_centroid - np.dot(source_centroid, new_rotation_matrix.T)

        # Check for convergence
        if np.linalg.norm(new_rotation_matrix - rotation_matrix) < tolerance and np.linalg.norm(new_translation_vector - translation_vector) < tolerance:
            break

        # Update transformation
        rotation_matrix = new_rotation_matrix
        translation_vector = new_translation_vector

    # Covert the rotation matrix to an angle in radians
    rotation = np.arctan2(R[1, 0], R[0, 0])

    return rotation, translation_vector


def rotation_matrix_to_angle(R):
    """
    Converts a 2D rotation matrix to an angle in radians.
    """
    # Extract the angle from the rotation matrix
    theta = np.arctan2(R[1, 0], R[0, 0])  # arctan2(sin(θ), cos(θ))
    return theta

# Beispiel-Punktwolken aus vorherigen Laserscans
laser_scan2 = np.random.uniform(1.0, 10.0, 180)  # Neuer Laserscan für einen weiteren Zeitschritt
point_cloud2 = polar_to_cartesian(laser_scan2)

# ICP anwenden, um die Transformation zwischen den Punktwolken zu berechnen
R, t = icp(point_cloud1, point_cloud2)
print("Rotation:\n", R)
print("Translation:\n", t)
