from uuid import uuid4

import numpy as np
from numpy import ndarray

from FastSLAM2.algorithms.hough_transformation import HoughTransformation
from FastSLAM2.algorithms.line_filter import LineFilter
from FastSLAM2.config import MAXIMUM_LANDMARK_DISTANCE
from FastSLAM2.models.landmark import Landmark
from FastSLAM2.models.measurement import Measurement
from FastSLAM2.utils.geometry_utils import GeometryUtils


class LandmarkUtils:
    """
    This utility class provides methods to extract landmarks from scanned points and associate them with known landmarks.
    """
    # List of known landmarks
    known_landmarks: list[Landmark] = []

    @staticmethod
    def get_measurements_to_landmarks(scanned_points: ndarray) -> list[Measurement]:
        """
        Extract landmarks from the given scanned points using hough transformation and DBSCAN clustering.
        A corner represents a landmark. The corners are calculated based on the intersection points of the scanned points.
        :param scanned_points: Scanned points in the form of a numpy array. The coordinates are also represented as numpy arrays [x, y].
        :return: Returns the measurements from the origin (0, 0) to the observed landmarks
        """
        # Get the observed landmarks
        observed_landmarks: list[Landmark] = LandmarkUtils.get_observed_landmarks(scanned_points)

        # Associate the observed landmarks with the existing landmarks. The IDs of the landmarks will be updated if a landmark is found.
        observed_landmarks = LandmarkUtils.__associate_landmarks(observed_landmarks)

        # Calculate the distance and angle of the corners to the origin (0, 0)
        measurements = []
        for landmark in observed_landmarks:
            dist, angle = GeometryUtils.calculate_distance_and_angle(landmark.x, landmark.y)
            measurements.append(Measurement(landmark.id, dist, angle))

        # Update the known landmarks with the updated landmarks
        LandmarkUtils.__update_known_landmarks(observed_landmarks)

        return measurements

    @staticmethod
    def get_observed_landmarks(scanned_points: ndarray) -> list[Landmark]:
        """
        Extract observed landmarks from the scanned points using the Hough transformation and DBSCAN clustering.
        A corner in the environment represents a landmark.
        :param scanned_points: The scanned points in the form of a numpy array. The coordinates are represented as numpy arrays [x, y].
        :return: Returns the observed landmarks
        """
        # Apply line filter to the scanned points to reduce noise. The filtered points are represented as arrays [x, y]
        filtered_points: ndarray = LineFilter.filter(scanned_points)

        # Detect line intersections in the filtered points
        intersection_points = HoughTransformation.detect_line_intersections(filtered_points)

        if len(intersection_points) > 0:
            # Cluster the intersection points to prevent multiple points for the same intersection
            # which can happen when multiple lines were detected for the same edge
            intersection_points = GeometryUtils.cluster_points(
                point_lists=intersection_points,
                eps=0.5,
                # Maximum distance between two samples  for one to be considered as in the neighborhood of the other
                min_samples=1  # The number of samples in a neighborhood for a point to be considered as a core point
            )

        # Get the corners which represent the landmarks
        return LandmarkUtils.__get_corners(intersection_points, filtered_points, threshold=0.2)

    @staticmethod
    def __get_corners(
            intersection_points: list[tuple[float, float]],
            scanned_points: ndarray,
            threshold=0.2
    ) -> list[Landmark]:
        """
        Search for corners in the environment based on the intersection points and the scanned points.
        :param intersection_points: The intersection points
        :return: Returns the corners as landmarks
        """
        corners: list[Landmark] = []
        for intersection_point in intersection_points:
            for scanned_point in scanned_points:
                # Calculate eucledeian distance between intersection point and scanned point.
                distance = np.sqrt(
                    (intersection_point[0] - scanned_point[0]) ** 2 + (intersection_point[1] - scanned_point[1]) ** 2)

                # If the distance is smaller than the threshold, the intersection point is a corner
                if distance <= threshold:
                    corners.append(Landmark(uuid4(), intersection_point[0], intersection_point[1]))
                    break  # Break the inner loop since the intersection point was already identified as a corner

        return corners

    @staticmethod
    def __associate_landmarks(observed_landmarks: list[Landmark]) -> list[Landmark]:
        """
        Search for associated landmarks based on the known landmarks and observed landmarks.
        If a landmark is found, the ID of the landmark will be referenced by the corresponding measurement.
        Else, the measurement will reference to a new landmark.
        :param observed_landmarks: The observed landmarks as points
        :return: Returns the updated measurements with the associated landmark IDs
        """
        # Search for associated landmarks.
        # If a landmark is found, the ID of the observed landmark will be overwritten with the ID of the known landmark.
        for i, observed_landmark in enumerate(observed_landmarks):
            for known_landmark in LandmarkUtils.known_landmarks:

                # Calculate the mahalanobis distance between the observed landmark and the particle's landmark
                # using the covariance matrix of the particle's landmark
                distance = GeometryUtils.mahalanobis_distance(
                    known_landmark.as_vector(),
                    observed_landmark.as_vector(),
                    known_landmark.cov
                )

                # If the distance from the observed landmark to an existing landmark is smaller than the threshold,
                # the landmark ID of the corresponding measurement will be overwritten with the ID of the associated landmark.
                if distance < MAXIMUM_LANDMARK_DISTANCE:
                    observed_landmark.landmark_id = known_landmark.id
                    break

        return observed_landmarks

    @staticmethod
    def __update_known_landmarks(observed_landmarks: list[Landmark]):
        """
        Update the list of known landmarks with the newly observed landmarks.
        :param observed_landmarks: The observed landmarks
        """
        # Get the IDs of the known landmarks
        known_landmark_ids = [landmark.id for landmark in LandmarkUtils.known_landmarks]

        # Filter the observed landmarks which are not already known
        new_landmarks = [landmark for landmark in observed_landmarks if landmark.id not in known_landmark_ids]

        # Extend the list of known landmarks with the new landmarks
        LandmarkUtils.known_landmarks.extend(new_landmarks)
