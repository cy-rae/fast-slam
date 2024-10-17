import math

import numpy as np
from numpy import ndarray
from sklearn.cluster import DBSCAN


class GeometryUtils:
    """
    This class contains utility methods for geometry calculations.
    """

    @staticmethod
    def mahalanobis_distance(position_a: ndarray, position_b: ndarray, covariance_matrix: ndarray) -> float:
        """
        Calculate the mahalanobis distance between two points A & B using the passed covariance matrix.
        :param position_a: The position of point A [x, y]
        :param position_b: The position of point B [x, y]
        :param covariance_matrix: The covariance matrix
        """
        delta = position_b - position_a
        distance = np.sqrt(delta.T @ np.linalg.inv(covariance_matrix) @ delta)
        return distance

    @staticmethod
    def cluster_points(
            point_lists: list[tuple[float, float]],
            eps: float,
            min_samples: int
    ) -> list[tuple[float, float]]:
        """
        Cluster the given points using DBSCAN.
        :param point_lists: The points to cluster represented as a list of tuples (x, y)
        :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other
        :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point
        :return: Returns the clustered points
        """
        # Convert the points to a numpy array
        points = np.array(point_lists)

        # Use DBSCAN to cluster the points
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

        # Extract the unique cluster labels
        labels = db.labels_
        unique_labels = set(labels)

        # Iterate through the unique clusters and collect their centroids
        cluster_centers: list[tuple[float, float]] = []
        for label in unique_labels:
            # -1 is the label for noise which can be ignored
            if label == -1:
                continue

            # Get the points which belong to the current cluster
            cluster_points: ndarray = points[labels == label]

            # Calculate centroids
            centroids: tuple[float, float] = cluster_points.mean(axis=0)
            cluster_centers.append(centroids)

        return cluster_centers

    @staticmethod
    def calculate_distance_and_angle(x: float, y: float) -> tuple[float, float]:
        """
        Calculate the distance and angle of a point to the origin (0, 0). The angle is rotated by -90 degrees.
        :param x: The x coordinate of the point
        :param y: The y coordinate of the point
        :return: Returns the distance(s) and angle(s) of the point(s) to the origin (0, 0)
        """
        distance: float = math.sqrt(x ** 2 + y ** 2)
        angle: float = math.atan2(y, x)
        return distance, angle
