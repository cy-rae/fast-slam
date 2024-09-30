import math

import numpy as np
from sklearn.cluster import DBSCAN

from src.models.directed_point import DirectedPoint
from src.models.laser_data import HAL
from src.models.measurement import Measurement
from src.models.point import Point


class Robot(DirectedPoint):
    """
    This class represents the robot
    """

    def __init__(self):
        super().__init__(0, 0, 0)

    @staticmethod
    def scan_environment() -> list[Point]:
        """
        Scan the environment using the laser data and return a list of points that were scanned by the laser.
        :return: Return a list of points that were scanned by the laser
        """
        # Get the laser data from the hardware abstraction layer
        laser_data = HAL.getLaserData()

        # Convert each laser data value to a point
        scanned_points: list[Point] = []
        for i in range(180):  # Laser data has 180 values
            # Extract the distance at index i
            dist = laser_data.values[i]

            # The final angle is centered (zeroed) at the front of the robot.
            angle = np.radians(i - 90)

            # Compute x, y coordinates from distance and angle
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)
            scanned_points.append(Point(x, y))
        return scanned_points

    def get_measurements_to_landmarks(self, scanned_points: list[Point], eps: float, min_samples: int) -> list[
        Measurement]:
        """
        Search for landmarks in passed list of points using distance-based clustering
        and measure their distances and angles to the robot. One cluster of points represents a landmark.
        :param scanned_points: The scanned obstacles as points.
        :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        :return: The distances and angles from the observed landmarks to the robot will be returned.
        """
        # Get scanned obstacles/points as vectors
        x_coords = [obstacle.x for obstacle in scanned_points]
        y_coords = [obstacle.y for obstacle in scanned_points]
        points = np.column_stack((x_coords, y_coords))

        # Use distance-based clustering to extract clusters which represent landmarks
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

        # Get the unique labels (-1, 0, 1, 2, ...)  which represent the clusters.
        labels: np.ndarray = db.labels_
        unique_labels: set[int] = set(labels)

        # Iterate over the clusters and calculate the distance and angle of the landmark to the robot
        measurements: list[Measurement] = []
        for label in unique_labels:
            #  The label -1 represents noise (isolated points) and can be skipped
            if label == -1:
                continue

            # Get the points which belong to the current cluster
            cluster_points = points[labels == label]

            # Calculate the centroid of the cluster
            x = np.mean(cluster_points[:, 0])
            y = np.mean(cluster_points[:, 1])

            # Calculate the distance and angle of the landmark to the current robot position
            dx = x - self.x
            dy = y - self.y
            distance = math.sqrt(dx ** 2 + dy ** 2)
            angle = math.atan2(dy, dx) - self.get_yaw_rad()

            # Create a new landmark object and add it to the landmarks list
            measurements.append(Measurement(distance, angle))

        return measurements
