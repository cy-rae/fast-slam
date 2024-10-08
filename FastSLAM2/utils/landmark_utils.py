import math

import numpy as np
from numpy import ndarray

from FastSLAM2.algorithms.hough_transformation import HoughTransformation
from FastSLAM2.algorithms.line_filter import LineFilter
from FastSLAM2.models.landmark import Landmark
from FastSLAM2.models.measurement import Measurement
from FastSLAM2.models.point import Point
from FastSLAM2.utils.geometry_utils import GeometryUtils


class LandmarkUtils:
    # Constants for the hough transformation
    __padding: int = 20
    __scale_factor: int = 100

    @staticmethod
    def get_measurements_to_landmarks(scanned_points: ndarray) -> tuple[list[Measurement], list[Point]]:
        """
        Extract landmarks from the given scanned points using hough transformation and DBSCAN clustering.
        A corner represents a landmark. The corners are calculated based on the intersection points of the scanned points.
        :param scanned_points: Scanned points in the form of a numpy array. The coordinates are also represented as numpy arrays [x, y].
        :return: Returns the extracted landmarks
        """
        # Apply line filter to the scanned points to reduce noise. The filtered points are represented as arrays [x, y]
        filtered_points: ndarray = LineFilter.filter(scanned_points)

        # Detect line intersections in the filtered points
        intersection_points = HoughTransformation.detect_line_intersections(filtered_points)

        # Cluster the intersection points to prevent multiple points for the same intersection
        # which can happen when multiple lines were detected for the same edge
        intersection_points = GeometryUtils.cluster_points(
            point_lists=intersection_points,
            eps=0.5,
            # Maximum distance between two samples  for one to be considered as in the neighborhood of the other
            min_samples=1  # The number of samples in a neighborhood for a point to be considered as a core point
        )

        # Get the corners which represent the landmarks
        corners: list[Point] = LandmarkUtils.__get_corners(intersection_points, filtered_points, threshold=0.2)

        # Calculate the distance and angle of the corners to the origin (0, 0)
        measurements = []
        for corner in corners:
            dist, angle = GeometryUtils.calculate_distance_and_angle(corner.x, corner.y)
            measurements.append(Measurement(dist, angle))

        return measurements, corners

    @staticmethod
    def __get_corners(
            intersection_points: list[tuple[float, float]],
            scanned_points: ndarray,
            threshold=0.2
    ) -> list[Point]:
        """
        Search for corners in the environment based on the intersection points and the scanned points.
        :param intersection_points: The intersection points
        :return: Returns the corners
        """
        corners: list[Point] = []
        for intersection_point in intersection_points:
            for scanned_point in scanned_points:
                # Calculate eucledeian distance between intersection point and scanned point.
                distance = np.sqrt(
                    (intersection_point[0] - scanned_point[0]) ** 2 + (intersection_point[1] - scanned_point[1]) ** 2)

                # If the distance is smaller than the threshold, the intersection point is a corner
                if distance <= threshold:
                    corners.append(Point(intersection_point[0], intersection_point[1]))
                    break  # Break the inner loop since the intersection point was already identified as a corner

        return corners

    @staticmethod
    def associate_landmarks(measurements: list[Measurement], observed_landmarks: list[Point]) -> list[Measurement]:
        """
        Search for associated landmarks based on the observed landmarks.
        If a landmark is found, the ID of the landmark will be referenced by the corresponding measurement.
        Else, the measurement will reference to a new landmark.
        :param measurements: The measurements to the observed landmarks. The landmark IDs are initially set to a new UUID.
        :param observed_landmarks: The observed landmarks as points
        :return: Returns the updated measurements with the associated landmark IDs
        """
        # Create a list to collect new landmark points
        new_landmarks: list[Landmark] = []

        for i, observed_landmark in enumerate(observed_landmarks):
            # Append the new landmark to the list
            new_landmark = Landmark(measurements[i].landmark_id, observed_landmark.x, observed_landmark.y)
            new_landmarks.append(new_landmark)

            # Search for associated landmarks. If a landmark is found, the ID will be overwritten
            for landmark in landmarks:
                # Calculate the mahalanobis distance between the observed landmark and the particle's landmark
                # using the covariance matrix of the particle's landmark
                distance = GeometryUtils.mahalanobis_distance(
                    landmark.as_vector(),
                    observed_landmark.as_vector(),
                    landmark.cov
                )

                # If the distance from the observed landmark to an existing landmark is smaller than the threshold,
                # the landmark ID of the corresponding measurement will be overwritten with the ID of the associated landmark.
                if distance < MAXIMUM_LANDMARK_DISTANCE:
                    measurements[i].landmark_id = landmark.id
                    new_landmarks.pop()  # Remove the new landmark since it is already associated with an existing landmark
                    break

        # Append the new landmarks to the existing landmarks
        landmarks.extend(new_landmarks)

        return measurements
