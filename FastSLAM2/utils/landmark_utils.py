import math

import cv2
import numpy as np
from numpy import ndarray
from scipy import ndimage
from sklearn.cluster import DBSCAN

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

        # Create hough transformation image
        image, width, height = LandmarkUtils.__create_hough_transformation_image(filtered_points)

        # Detect lines using hough transformation
        lines = LandmarkUtils.__hough_line_detection(image)

        # Calculate the intersection points. If no intersection points are found, return empty lists
        intersection_points = LandmarkUtils.__calculate_intersections(lines, width, height)
        if len(intersection_points) == 0:
            return [], []

        # Cluster the intersection points to prevent multiple points for the same intersection
        # which can happen when multiple lines were detected for the same edge
        intersection_points = LandmarkUtils.__cluster_points(intersection_points, 10, 1)

        # Convert the intersection points back to the original coordinate space
        intersection_points = LandmarkUtils.__convert_back_to_original_space(filtered_points, intersection_points)

        # Get the corners which represent the landmarks
        corners: list[Point] = LandmarkUtils.__get_corners(intersection_points, filtered_points, threshold=0.2)

        # Calculate the distance and angle of the corners to the origin (0, 0)
        measurements = []
        for corner in corners:
            dist, angle = LandmarkUtils.__calculate_distance_and_angle(corner.x, corner.y)
            measurements.append(Measurement(dist, angle))

        return measurements, corners

    @staticmethod
    def __create_hough_transformation_image(scanned_points: np.ndarray):
        """
        Create an image for the hough transformation with the scanned points.
        :param scanned_points: The scanned points
        :return: Returns the image for the hough transformation
        """
        # Get the scaled min and max values of the scanned points
        min_x = int(np.min(scanned_points[:, 0] * LandmarkUtils.__scale_factor))
        min_y = int(np.min(scanned_points[:, 1] * LandmarkUtils.__scale_factor))
        max_x = int(np.max(scanned_points[:, 0] * LandmarkUtils.__scale_factor))
        max_y = int(np.max(scanned_points[:, 1] * LandmarkUtils.__scale_factor))

        # Calculate the offset to bring all points into the positive coordinate system for the transformation
        offset_x = -min_x if min_x < 0 else 0
        offset_y = -min_y if min_y < 0 else 0
        offset_x += LandmarkUtils.__padding  # Apply padding to avoid drawing points at the edge of the image
        offset_y += LandmarkUtils.__padding

        # Create a new image for the transformation with the offsets
        width = max_x + offset_x + LandmarkUtils.__padding
        height = max_y + offset_y + LandmarkUtils.__padding
        image = np.zeros((height, width), dtype=np.uint8)

        # Scale and add the scanned points to the image as circles
        for point in scanned_points:
            x = int(point[0] * LandmarkUtils.__scale_factor) + offset_x
            y = int(point[1] * LandmarkUtils.__scale_factor) + offset_y
            cv2.circle(image, center=(x, y), radius=2, color=255, thickness=-1)

        return image, width, height

    @staticmethod
    def __hough_line_detection(image):
        """
        Detect lines in the given image using the hough transformation.
        :param image: The image to detect lines in
        :return: Returns the detected lines
        """
        # Schritt 4: Kantenextraktion mit Canny
        edges = cv2.Canny(image, 100, 150, apertureSize=3)

        # Schritt 5: Verwende die Hough-Transformation zur Linienerkennung
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 90)

        return lines

    @staticmethod
    def __calculate_intersections(lines, width, height) -> list[tuple[float, float]]:
        """
        Calculate the intersection points of the given lines.
        :param lines: The lines to calculate the intersection points for
        :param width: The width of the image
        :param height: The height of the image
        :return: Returns the intersection points
        """
        # Check if no lines were detected
        if lines is None:
            return []

        # Calculate the intersection points of the lines
        intersections: list[tuple[float, float]] = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                # Get the rho and theta values of the lines
                rho1, theta1 = lines[i][0]
                rho2, theta2 = lines[j][0]

                # Calculate the angle difference between the lines
                angle_diff: float = abs(theta1 - theta2)
                angle_diff: float = min(angle_diff, np.pi - angle_diff)  # Normalize the angle difference to [0, pi]

                # If the angle difference is too small, the lines are almost parallel and the intersection point will be ignored
                if angle_diff < np.deg2rad(45):
                    continue

                # Calculate the coefficients of the lines
                a1, b1 = np.cos(theta1), np.sin(theta1)
                a2, b2 = np.cos(theta2), np.sin(theta2)

                # Calculate the determinant of the lines to check if they intersect
                determinant: float = a1 * b2 - a2 * b1
                if abs(determinant) > 1e-10:
                    # Calculate the intersection point
                    x: float = (b2 * rho1 - b1 * rho2) / determinant
                    y: float = (a1 * rho2 - a2 * rho1) / determinant

                    # Only consider intersection points within the image bounds
                    if 0 <= x < width and 0 <= y < height:
                        intersections.append((x, y))

        return intersections

    @staticmethod
    def __cluster_points(point_list: list[tuple[float, float]], eps=10, min_samples=1) -> list[tuple[float, float]]:
        """
        Cluster the given points using DBSCAN.
        :param point_list: The points to cluster
        :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other
        :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point
        :return: Returns the clustered points
        """
        # Convert the points to a numpy array
        points: ndarray = np.array(point_list)

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
    def __convert_back_to_original_space(scanned_points, cluster_centers):
        """
        Convert the clustered points back to the original coordinate space.
        :param scanned_points: The scanned points
        :param cluster_centers: The clustered points
        :return:
        """
        original_points: list[tuple[float, float]] = []

        # Calculate the offset to move all points into the correct position of the coordinate system
        min_x = int(np.min(scanned_points[:, 0] * LandmarkUtils.__scale_factor))
        min_y = int(np.min(scanned_points[:, 1] * LandmarkUtils.__scale_factor))
        offset_x = -min_x if min_x < 0 else 0
        offset_y = -min_y if min_y < 0 else 0
        offset_x += LandmarkUtils.__padding
        offset_y += LandmarkUtils.__padding

        # Calculate the original points
        for x, y in cluster_centers:
            original_x = (x - offset_x) / LandmarkUtils.__scale_factor
            original_y = (y - offset_y) / LandmarkUtils.__scale_factor
            original_points.append((original_x, original_y))

        return original_points

    @staticmethod
    def __calculate_distance_and_angle(x: float, y: float):
        """
        Calculate the distance and angle of a point to the origin (0, 0). The angle is rotated by -90 degrees.
        :param x: The x coordinate of the point
        :param y: The y coordinate of the point
        :return: Returns the distance(s) and angle(s) of the point(s) to the origin (0, 0)
        """
        distance = math.sqrt(x ** 2 + y ** 2)
        angle = math.atan2(y, x)
        return distance, angle

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