import numpy as np
from numpy import ndarray

from fast_slam_2.algorithms.hough_transformation import HoughTransformation
from fast_slam_2.algorithms.line_filter import LineFilter
from fast_slam_2.config import MAXIMUM_LANDMARK_DISTANCE
from fast_slam_2.models.landmark import Landmark
from fast_slam_2.models.measurement import Measurement
from fast_slam_2.models.particle import Particle
from fast_slam_2.utils.geometry_utils import GeometryUtils


class LandmarkUtils:
    """
    This utility class provides methods to extract landmarks from scanned points and associate them with known landmarks.
    """
    # List of known landmarks
    known_landmarks: list[Landmark] = []

    @staticmethod
    def get_measurements_to_landmarks(scanned_points: ndarray) -> list[Measurement]:
        """
        Extract landmarks from the given scanned points using hough transformation and DBSCAN clustering. A corner
        represents a landmark. The corners are calculated based on the intersection points of the scanned points.
        :param scanned_points: Scanned points as a Nx2 array.
        :return: Returns the measurements from the origin (0, 0) to the observed landmarks
        """
        # Get the observed landmarks
        observed_landmarks: list[Landmark] = LandmarkUtils.get_observed_landmarks(scanned_points)

        # Calculate the distance and angle of the corners to the origin (0, 0)
        measurements = []
        for landmark in observed_landmarks:
            dist, angle = GeometryUtils.calculate_distance_and_angle(landmark.x, landmark.y)
            measurements.append(Measurement(dist, angle))

        return measurements

    @staticmethod
    def get_observed_landmarks(scanned_points: ndarray) -> list[Landmark]:
        """
        Extract observed landmarks from the scanned points using the Hough transformation and DBSCAN clustering.
        A corner in the environment represents a landmark.
        :param scanned_points: The scanned points in as a Nx2 array.
        :return: Returns the observed landmarks
        """
        # Apply line filter to the scanned points to reduce noise. The filtered points are represented as arrays [x, y]
        filtered_points: ndarray = LineFilter.filter(scanned_points)

        # Detect line intersections in the filtered points
        intersection_points: list[tuple[float, float]] = HoughTransformation.detect_line_intersections(filtered_points)

        if len(intersection_points) > 0:
            # Cluster the intersection points to prevent multiple points for the same intersection
            # which can happen when multiple lines were detected for the same edge
            intersection_points = GeometryUtils.cluster_points(
                point_lists=intersection_points,
                eps=0.5,  # Maximum distance between two samples for one to be considered as a part of the other
                min_samples=1  # The number of samples in a neighborhood for a point to be considered as a core point
            )

        # Get the corners which represent the landmarks TODO
        # return LandmarkUtils.__get_corners(intersection_points, filtered_points, threshold=0.1)

        # Convert the intersection points to landmarks
        landmarks = []
        for intersection_point in intersection_points:
            landmarks.append(Landmark(intersection_point[0], intersection_point[1], np.array([[0.1, 0], [0, 0.1]])))

        return landmarks

    @staticmethod
    def __get_corners(
            intersection_points: list[tuple[float, float]],
            scanned_points: ndarray,
            threshold
    ) -> list[Landmark]:
        """
        Search for corners in the environment based on the intersection points and the scanned points.
        :param intersection_points: The intersection points
        :return: Returns the corners as landmarks
        """
        corners: list[Landmark] = []
        for intersection_point in intersection_points:
            for scanned_point in scanned_points:
                # Calculate euclidian distance between intersection point and scanned point.
                distance = np.sqrt(
                    (intersection_point[0] - scanned_point[0]) ** 2 + (intersection_point[1] - scanned_point[1]) ** 2
                )

                # If the distance is smaller than the threshold, the intersection point is a corner
                if distance <= threshold:
                    corners.append(Landmark(intersection_point[0], intersection_point[1]))
                    break  # Break the inner loop since the intersection point was already identified as a corner

        return corners

    @staticmethod
    def associate_landmarks(
            observed_landmark: Landmark,
            particle_landmarks: list[Landmark]
    ) -> tuple[Landmark or None, int or None]:
        """
        Search for associated landmarks based on the known landmarks and observed landmarks.
        :param observed_landmark: The observed landmark as a point
        :param particle_landmarks: The landmarks of the particle
        :return: Returns the associated landmark and its index
        """
        # Search for associated landmarks. Iterating through all landmarks
        for i, known_landmark in enumerate(particle_landmarks):
            # Calculate the mahalanobis distance between the observed landmark and the particle's landmark
            # using the covariance matrix of the particle's landmark
            distance = GeometryUtils.mahalanobis_distance(
                known_landmark.as_vector(),
                observed_landmark.as_vector(),
                known_landmark.cov
            )

            # If the distance from the observed landmark to an existing landmark is smaller than the threshold,
            # the landmark ID of the observed landmark will be overwritten with the ID of the associated landmark.
            if distance < MAXIMUM_LANDMARK_DISTANCE:
                return known_landmark, i

        return None, None

    @staticmethod
    def update_known_landmarks(particles: list[Particle]):
        """
        Update the known landmarks by clustering the particle's landmarks
        """
        # Get all landmarks as Nx2 array
        all_landmarks: list[tuple[float, float]] = []
        for particle in particles:
            for landmark in particle.landmarks:
                all_landmarks.append((landmark.x, landmark.y))

        # Get the minimum number of landmarks to determine a cluster which is 70% of the average number of landmarks per particle
        avg_landmarks = len(all_landmarks) / len(particles)
        min_samples = int(avg_landmarks * 0.7)

        # Skip clustering if there are not enough landmarks
        if min_samples < 1:
            return

        # Cluster the landmarks
        clustered_landmarks = GeometryUtils.cluster_points(all_landmarks, eps=0.5, min_samples=min_samples)

        # Update the known landmarks
        LandmarkUtils.known_landmarks = []
        for landmark in clustered_landmarks:
            LandmarkUtils.known_landmarks.append(Landmark(landmark[0], landmark[1]))
