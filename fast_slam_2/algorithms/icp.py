import numpy as np
from scipy.spatial import KDTree


class ICP:
    @staticmethod
    def get_transformation(
            source_points: np.ndarray,
            target_points: np.ndarray,
            max_iterations=100,
            threshold=1e-5
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the best fitting rotation matrix and translation vector that aligns the source points to the target points.
        :param source_points: Nx2 array of source points
        :param target_points: Nx2 array of target points
        :param max_iterations: Maximum number of iterations. Default is 100
        :param threshold: Tolerance threshold for convergence. Default is 1e-5.
        """
        # Initialize the previous error to a large value to ensure the loop runs at least once
        prev_error = float('inf')

        # Initialize the rotation matrix and translation vector
        total_rotation = np.eye(2)
        total_translation = np.zeros((2,))

        for i in range(max_iterations):
            # Get the nearest neighbors in the target point cloud
            tree = KDTree(target_points)
            distances, indices = tree.query(source_points)

            # Calculate the transformation between the source and target points
            rotation, translation = ICP.best_fit_transform(source_points, target_points[indices])

            # Apply the transformation to the source points
            source_points = np.dot(source_points, rotation.T) + translation

            # Update the overall transformation (combining rotations and translations)
            total_rotation = np.dot(rotation, total_rotation)
            total_translation = np.dot(rotation, total_translation) + translation

            # Calculate the mean error and break the loop if the error is less than the threshold
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < threshold:
                break

            # Set previous error for next iteration
            prev_error = mean_error

        return total_rotation, total_translation

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
        cov = np.dot(centered_source.T, centered_target)

        # Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(cov)

        # Compute the rotation matrix
        rotation_matrix = np.dot(Vt.T, U.T)

        # Handle reflection case (det(R) should be 1, if det(R) == -1, we correct it)
        if np.linalg.det(rotation_matrix) < 0:
            Vt[-1, :] *= -1
            rotation_matrix = np.dot(Vt.T, U.T)

        # Compute the translation vector
        translation_vector = centroid_target - np.dot(rotation_matrix, centroid_source)

        return rotation_matrix, translation_vector
