import cv2
import numpy as np


class HoughTransformation:
    # Constants for the hough transformation
    __padding: int = 20
    __scale_factor: int = 100

    @staticmethod
    def detect_line_intersections(points: np.ndarray):
        """
        Detect line intersections in the given points using the hough transformation.
        :param points: The points to detect line intersections in. The points are represented as a Nx2 array
        :return: Returns the intersection points
        """
        # Create hough transformation image
        image, width, height = HoughTransformation.__create_hough_transformation_image(points)

        # Detect lines using hough transformation
        lines = HoughTransformation.__detect_lines(image)

        # Calculate the intersection points. If no intersection points are found, return empty lists
        intersection_points = HoughTransformation.__calculate_intersections(lines, width, height)
        if len(intersection_points) == 0:
            return [], []

        # Convert the intersection points back to the original coordinate space
        intersection_points = HoughTransformation.__convert_back_to_original_space(points, intersection_points)

        return intersection_points

    @staticmethod
    def __create_hough_transformation_image(scanned_points: np.ndarray):
        """
        Create an image for the hough transformation with the scanned points.
        :param scanned_points: The scanned points
        :return: Returns the image for the hough transformation
        """
        # Get the scaled min and max values of the scanned points
        min_x = int(np.min(scanned_points[:, 0] * HoughTransformation.__scale_factor))
        min_y = int(np.min(scanned_points[:, 1] * HoughTransformation.__scale_factor))
        max_x = int(np.max(scanned_points[:, 0] * HoughTransformation.__scale_factor))
        max_y = int(np.max(scanned_points[:, 1] * HoughTransformation.__scale_factor))

        # Calculate the offset to bring all points into the positive coordinate system for the transformation
        offset_x = -min_x if min_x < 0 else 0
        offset_y = -min_y if min_y < 0 else 0
        offset_x += HoughTransformation.__padding  # Apply padding to avoid drawing points at the edge of the image
        offset_y += HoughTransformation.__padding

        # Create a new image for the transformation with the offsets
        width = max_x + offset_x + HoughTransformation.__padding
        height = max_y + offset_y + HoughTransformation.__padding
        image = np.zeros((height, width), dtype=np.uint8)

        # Scale and add the scanned points to the image as circles
        for point in scanned_points:
            x = int(point[0] * HoughTransformation.__scale_factor) + offset_x
            y = int(point[1] * HoughTransformation.__scale_factor) + offset_y
            cv2.circle(image, center=(x, y), radius=2, color=255, thickness=-1)

        return image, width, height


    @staticmethod
    def __detect_lines(image):
        """
        Detect lines in the given image using the hough transformation.
        :param image: The image to detect lines in
        :return: Returns the detected lines
        """
        # Extract edges from the image using the Canny edge detector
        edges = cv2.Canny(image, 100, 150, apertureSize=3)

        # Use hough transformation to detect lines in the image
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
    def __convert_back_to_original_space(scanned_points, cluster_centers):
        """
        Convert the clustered points back to the original coordinate space.
        :param scanned_points: The scanned points
        :param cluster_centers: The clustered points
        :return:
        """
        original_points: list[tuple[float, float]] = []

        # Calculate the offset to move all points into the correct position of the coordinate system
        min_x = int(np.min(scanned_points[:, 0] * HoughTransformation.__scale_factor))
        min_y = int(np.min(scanned_points[:, 1] * HoughTransformation.__scale_factor))
        offset_x = -min_x if min_x < 0 else 0
        offset_y = -min_y if min_y < 0 else 0
        offset_x += HoughTransformation.__padding
        offset_y += HoughTransformation.__padding

        # Calculate the original points
        for x, y in cluster_centers:
            original_x = (x - offset_x) / HoughTransformation.__scale_factor
            original_y = (y - offset_y) / HoughTransformation.__scale_factor
            original_points.append((original_x, original_y))

        return original_points