import numpy as np
from numpy import ndarray
from scipy.spatial import KDTree


class ICP:
    @staticmethod
    def run(
            source_points: ndarray,
            target_points: ndarray,
            max_iterations=300,
            tolerance=1e-6
    ) -> tuple[ndarray, ndarray]:
        """
        Iterative closest point algorithm for 2D point clouds to find the optimal rotation and translation between them.
        :param source_points: Nx2 array of source points
        :param target_points: Nx2 array of target points
        :param max_iterations: Maximum number of iterations
        :param tolerance: Tolerance for convergence
        :return: Returns the rotation in radians and translation vector
        """
        # Initial transformation with identity matrix and zero vector
        translation_vector = np.zeros((2,))
        rotation_matrix = np.eye(2)

        for _ in range(max_iterations):
            # Apply the current transformation to source points
            transformed_points = np.dot(source_points, rotation_matrix.T) + translation_vector

            # Find nearest neighbors in target point cloud
            tree = KDTree(target_points)
            _, indices = tree.query(transformed_points)

            # Corresponding points from target
            matched_points = target_points[indices]
            matched_points = matched_points.reshape(-1, 2)  # Reshape to Nx2 array

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
            if np.linalg.norm(new_rotation_matrix - rotation_matrix) < tolerance and np.linalg.norm(
                    new_translation_vector - translation_vector) < tolerance:
                break

            # Update transformation
            translation_vector = new_translation_vector
            rotation_matrix = new_rotation_matrix

        return translation_vector, rotation_matrix

    @staticmethod
    def best_fit_transform(source_points, target_points):
        """
        Compute the best fitting rotation matrix and translation vector
        that aligns the source points to the target points.
        """
        # Compute centroids of both point sets
        centroid_source = np.mean(source_points, axis=0)
        centroid_target = np.mean(target_points, axis=0)

        # Center the points
        centered_source = source_points - centroid_source
        centered_target = target_points - centroid_target

        # Compute covariance matrix
        H = centered_source.T @ centered_target

        # Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(H)

        # Compute the rotation matrix
        R = Vt.T @ U.T

        # Handle reflection case (det(R) should be 1, if det(R) == -1, we correct it)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute the translation vector
        t = centroid_target.T - R @ centroid_source.T

        return R, t