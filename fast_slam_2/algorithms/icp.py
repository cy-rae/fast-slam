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

        return rotation_matrix, translation_vector

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
        t = centroid_target - np.dot(rotation_matrix, centroid_source)

        return rotation_matrix, t

    @staticmethod
    def get_transformation(
            source_points: ndarray,
            target_points: ndarray,
            max_iterations=100,
            threshold=1e-5
    ) -> tuple[ndarray, ndarray]:
        """
        Compute the best fitting rotation matrix and translation vector that aligns the source points to the target points.
        :param source_points: Nx2 array of source points
        :param target_points: Nx2 array of target points
        :param max_iterations: Maximum number of iterations. Default is 50
        :param threshold: Tolerance threshold for convergence. Default is 1e-5.
        """
        # Initialize the previous error to a large value to ensure the loop runs at least once
        prev_error = 1

        # Initialize the rotation matrix and translation vector
        rotation_matrix = np.eye(2)
        translation_vector = np.zeros((2,))

        for i in range(max_iterations):
            # Get the nearest neighbors in the target point cloud
            tree = KDTree(target_points)
            distances, indices = tree.query(source_points)

            # Calculate the transformation between the source and target points
            rotation_matrix, translation_vector = ICP.best_fit_transform(source_points, target_points[indices])

            # Apply the transformation to the source points
            source_points = np.dot(source_points, rotation_matrix.T) + translation_vector

            # Calculate the mean error and break the loop if the error is less than the threshold
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < threshold:
                break

            # Set previous error for next iteration
            prev_error = mean_error

        return rotation_matrix, translation_vector

