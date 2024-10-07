"""
This script contains all the classes and methods that are needed to run the FastSLAM 2.0 algorithm.
You can upload this script to the JDE Robots platform and run it in the simulation environment.
The script will create a map with the robot, particles, landmarks, and obstacles.
The robot will move in the environment and update its position based on the observed landmarks and the FastSLAM 2.0 algorithm.
"""
import math
import random
import uuid

import HAL
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from numpy import ndarray
from scipy import ndimage
from scipy.stats import multivariate_normal
from sklearn.cluster import DBSCAN


# region Models
class Point:
    """
    Class to represent a point in 2D space.
    """

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def as_vector(self):
        """
        Get the pose/mean of the point as a vector [x, y].
        :return: Returns the position of the point as a numpy array [x, y]
        """
        return np.array([self.x, self.y])

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


class DirectedPoint(Point):
    """
    Class to represent a point in 2D space with a yaw value / angle in degrees.
    """

    def __init__(self, x: float, y: float, yaw: float):
        super().__init__(x, y)
        self.yaw = yaw


class Measurement:
    """
    Class to represent the measurements of an observed landmark (distance and angle in radians).
    """

    def __init__(self, distance: float, yaw: float):
        self.landmark_id: uuid.UUID = uuid.uuid4()
        self.distance = distance
        self.yaw = yaw

    def as_vector(self):
        return np.array([self.distance, self.yaw])


class Landmark(Point):
    """
    Class to represent a landmark in 2D space.
    A landmark has a covariance matrix which describes the uncertainty of the landmark's position.
    """

    def __init__(self, identifier: uuid.UUID, x: float, y: float, cov: ndarray = np.array([[0.1, 0], [0, 0.1]])):
        super().__init__(x, y)
        self.id = identifier
        self.cov = cov


class Particle(DirectedPoint):
    """
    Class to represent a particle in the FastSLAM 2.0 algorithm.
    """

    def __init__(self, x: float, y: float, yaw: float):
        super().__init__(x, y, yaw)
        self.weight = 1.0 / NUM_PARTICLES
        self.landmarks: list[Landmark] = []

    def get_index_of_landmark(self, landmark_id: uuid.UUID) -> int or None:
        """
        Get the index of the landmark with the passed ID.
        :param landmark_id: The ID of the landmark
        :return: Returns the index of the landmark or None if the landmark is not found
        """
        for i, landmark in enumerate(self.landmarks):
            if landmark.id == landmark_id:
                return i
        return None


class Robot(DirectedPoint):
    """
    This class represents the robot
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        super().__init__(x, y, yaw)
        self.x_prev = 0.0
        self.y_prev = 0.0
        self.yaw_prev = 0.0
        self.timestamp_prev = 0.0
        self.x_prev = 0.0
        self.y_prev = 0.0
        self.yaw_prev = 0.0

    @staticmethod
    def scan_environment() -> list[Point]:
        """
        Scan the environment using the laser data and return a list of points that were scanned by the laser.
        :return: Return a list of points that were scanned by the laser
        """
        # Get laser data from the robot. Laser data contains the distances and angles to obstacles in the environment.
        laser_data = HAL.getLaserData()

        # Convert each laser data value to a point
        scanned_points: list[Point] = []
        for i in range(180):  # Laser data has 180 values
            # Extract the distance at index i
            dist = laser_data.values[i]

            # Skip invalid distances (e.g., min or max range)
            if dist < laser_data.minRange or dist > laser_data.maxRange:
                continue

            # The final angle is centered (zeroed) at the front of the robot.
            angle = np.radians(i - 90)

            # Compute x, y coordinates from distance and angle
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)
            scanned_points.append(Point(x, y))
        return scanned_points

    def move(self):
        """
        Move the robot based on the bumper state.
        :return: Returns the linear and angular velocity of the robot
        """
        # First, move robot in real world
        # Set linear and angular velocity depending on the bumper state.
        bumper_state = HAL.getBumperData().state
        if bumper_state == 1:
            # If the robot hits the wall, the linear velocity will be set to 0
            v = 0

            # If the robot hits the wall, the angular velocity will be set depending on the bumper that was hit
            bumper = HAL.getBumperData().bumper
            if bumper == 0:  # right bumper
                w = 1
            else:  # left or center bumper
                w = -1

        # If the robot does not hit the wall, the linear and angular velocities will be set to 1 and 0 respectively
        else:
            v = 1
            w = 0

        # Move robot
        HAL.setV(v)
        HAL.setW(w)

        # Second, calculate delta values for odometry simulation
        return self.__simulate_odometry()

    def __simulate_odometry(self) -> tuple[float, float]:
        """
        Simulate odometry data by calculating the difference in x and y coordinates and the difference in yaw angle (in radians) to the previous pose.
        Since there is no odometry data, we have to calculate the delta values based on the current pose and the previous pose.
        :return: Returns the difference in x and y coordinates and the difference in yaw angle (in radians) to the previous pose.
        """
        # Get current pose
        x_curr = HAL.getPose3d().x
        y_curr = HAL.getPose3d().y
        yaw_curr = HAL.getPose3d().yaw  # in radians

        # Calculate linear and angular velocity
        v = self.__calculate_linear_delta(x_curr, y_curr)
        w = yaw_curr - self.yaw_prev

        # Update previous values
        self.x_prev = x_curr
        self.y_prev = y_curr
        self.yaw_prev = yaw_curr

        return v, w

    def __calculate_linear_delta(self, x_curr: float, y_curr: float):
        """
        Calculate the linear velocity of the robot based on the current and previous pose.
        :param x_curr: Current x-coordinate
        :param y_curr: Current y-coordinate
        :return: Returns the linear velocity of the robot
        """
        # Calculate the change in x and y
        delta_x = x_curr - self.x_prev
        delta_y = y_curr - self.y_prev
        delta_d = np.sqrt(delta_x ** 2 + delta_y ** 2)

        return delta_d


# endregion

# region Services
class LandmarkService:
    # Constants for the hough transformation
    __padding: int = 20
    __scale_factor: int = 100

    @staticmethod
    def get_measurements_to_landmarks(scanned_points: list[Point]) -> tuple[list[Measurement], list[Point]]:
        """
        Extract landmarks from the given scanned points using hough transformation and DBSCAN clustering.
        A corner represents a landmark. The corners are calculated based on the intersection points of the scanned points.
        :param scanned_points: Scanned points in the form of a numpy array
        :return: Returns the extracted landmarks
        """
        # Apply line filter to the scanned points to reduce noise. The filtered points are represented as a 2D array of float tuples
        point_data = np.array([point.as_vector() for point in scanned_points])
        filtered_points: ndarray = LandmarkService.line_filter(point_data)

        # Create hough transformation image
        image, width, height = LandmarkService.__create_hough_transformation_image(filtered_points)

        # Detect lines using hough transformation
        lines = LandmarkService.__hough_line_detection(image)

        # Calculate the intersection points. If no intersection points are found, return empty lists
        intersection_points = LandmarkService.__calculate_intersections(lines, width, height)
        if len(intersection_points) == 0:
            return [], []

        # Cluster the intersection points to prevent multiple points for the same intersection
        # which can happen when multiple lines were detected for the same edge
        intersection_points = LandmarkService.__cluster_points(intersection_points, 10, 1)

        # Convert the intersection points back to the original coordinate space
        intersection_points = LandmarkService.__convert_back_to_original_space(filtered_points, intersection_points)

        # Get the corners which represent the landmarks
        corners: list[Point] = LandmarkService.__get_corners(intersection_points, filtered_points, threshold=0.2)

        # Calculate the distance and angle of the corners to the origin (0, 0)
        measurements = []
        for corner in corners:
            dist, angle = LandmarkService.__calculate_distance_and_angle(corner.x, corner.y)
            measurements.append(Measurement(dist, angle))

        return measurements, corners

    @staticmethod
    def line_filter(points, sigma=0.1):
        """
        Apply a Gaussian filter to the points to reduce noise.
        :param points: The points to filter
        :param sigma: The standard deviation of the Gaussian filter
        :return: Returns the filtered points
        """
        x_filtered = ndimage.gaussian_filter1d(points[:, 0], sigma=sigma)
        y_filtered = ndimage.gaussian_filter1d(points[:, 1], sigma=sigma)
        return np.vstack((x_filtered, y_filtered)).T

    @staticmethod
    def __create_hough_transformation_image(scanned_points: np.ndarray):
        """
        Create an image for the hough transformation with the scanned points.
        :param scanned_points: The scanned points
        :return: Returns the image for the hough transformation
        """
        # Get the scaled min and max values of the scanned points
        min_x = int(np.min(scanned_points[:, 0] * LandmarkService.__scale_factor))
        min_y = int(np.min(scanned_points[:, 1] * LandmarkService.__scale_factor))
        max_x = int(np.max(scanned_points[:, 0] * LandmarkService.__scale_factor))
        max_y = int(np.max(scanned_points[:, 1] * LandmarkService.__scale_factor))

        # Calculate the offset to bring all points into the positive coordinate system for the transformation
        offset_x = -min_x if min_x < 0 else 0
        offset_y = -min_y if min_y < 0 else 0
        offset_x += LandmarkService.__padding  # Apply padding to avoid drawing points at the edge of the image
        offset_y += LandmarkService.__padding

        # Create a new image for the transformation with the offsets
        width = max_x + offset_x + LandmarkService.__padding
        height = max_y + offset_y + LandmarkService.__padding
        image = np.zeros((height, width), dtype=np.uint8)

        # Scale and add the scanned points to the image as circles
        for point in scanned_points:
            x = int(point[0] * LandmarkService.__scale_factor) + offset_x
            y = int(point[1] * LandmarkService.__scale_factor) + offset_y
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
        min_x = int(np.min(scanned_points[:, 0] * LandmarkService.__scale_factor))
        min_y = int(np.min(scanned_points[:, 1] * LandmarkService.__scale_factor))
        offset_x = -min_x if min_x < 0 else 0
        offset_y = -min_y if min_y < 0 else 0
        offset_x += LandmarkService.__padding
        offset_y += LandmarkService.__padding

        # Calculate the original points
        for x, y in cluster_centers:
            original_x = (x - offset_x) / LandmarkService.__scale_factor
            original_y = (y - offset_y) / LandmarkService.__scale_factor
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
                distance = Point.mahalanobis_distance(
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


class InterpretationService:
    """
    Service class to interpret the results of the FastSLAM 2.0 algorithm.
    """

    @staticmethod
    def update_obstacles(scanned_obstacles: list[Point]):
        """
        Filter out the new obstacles which will be added to the obstacles list so the map will show new borders and obstacles.
        :param scanned_obstacles: The scanned obstacles
        """
        # Apply translation and rotation of the robot to the scanned obstacles to get the global coordinates
        global_points: list[Point] = []
        for scanned_obstacle in scanned_obstacles:
            x_global = robot.x + (
                    scanned_obstacle.x * np.cos(robot.yaw) - scanned_obstacle.y * np.sin(robot.yaw))
            y_global = robot.y + (
                    scanned_obstacle.x * np.sin(robot.yaw) + scanned_obstacle.y * np.cos(robot.yaw))
            global_points.append(Point(x_global, y_global))

        # Round the coordinates of the scanned obstacles to 2 decimal places to add noise to the data.
        # This is important since the laser data is not 100% accurate.
        # Thus, no new obstacles will be added to the same position when scanning the same obstacle multiple times.
        for global_point in global_points:
            global_point.x = round(global_point.x, 1)
            global_point.y = round(global_point.y, 1)

        # Update obstacles list with the scanned obstacles
        existing_coords = {(obstacle.x, obstacle.y) for obstacle in obstacles}
        new_obstacles = [obstacle for obstacle in global_points if
                         (obstacle.x, obstacle.y) not in existing_coords]
        obstacles.extend(new_obstacles)

        return obstacles

    @staticmethod
    def estimate_robot_position(particles: list[Particle]) -> tuple[float, float, float]:
        """
        Calculate the estimated position of the robot based on the passed particles.
        The estimation is based on the mean of the particles.
        :param particles: The particles which represent the possible positions of the robot.
        :return: Returns the estimated position of the robot as a tuple (x, y, yaw)
        """
        x_mean = 0.0
        y_mean = 0.0
        yaw_mean = 0.0
        total_weight = sum(p.weight for p in particles)
        # print(total_weight)

        # Calculate the mean of the particles
        for p in particles:
            x_mean += p.x * p.weight
            y_mean += p.y * p.weight
            yaw_mean += p.yaw * p.weight

        # Normalize the estimated position
        x_mean /= total_weight
        y_mean /= total_weight
        yaw_mean /= total_weight
        yaw_mean = (yaw_mean + np.pi) % (2 * np.pi) - np.pi  # Ensure yaw is between -pi and pi

        return x_mean, y_mean, yaw_mean


class MapService:
    """
    Service class to plot the map with the robot, particles, landmarks and obstacles/borders.
    """

    @staticmethod
    def plot_map():
        """
        Plot the map with the robot, particles, landmarks and obstacles/borders.
        """
        try:
            image, draw = MapService.__init_plot()
            MapService.__plot_as_arrows(draw, directed_points=[robot], scale=10,
                                        color='red')  # Plot the robot as a red arrow
            MapService.__plot_as_arrows(draw, directed_points=fast_slam.particles, scale=7,
                                        color='blue')  # Plot the particles as blue arrows
            MapService.__plot_as_dots(draw, obstacles, 'black')  # Mark obstacles as black dots
            MapService.__plot_as_dots(draw, landmarks, 'green')  # Mark landmarks as green dots

            # Save the plot as an image file
            image.save('/usr/share/nginx/html/images/map.jpg', 'JPEG')
        except Exception as e:
            print(e)

    @staticmethod
    def __init_plot():
        """
        Initialize the plot
        """
        # Image size and background color
        width, height = 1500, 1500
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        # Draw the axes
        center_x = width // 2
        center_y = height // 2
        draw.line((0, center_y, width, center_y), fill="black", width=2)  # X-Achse
        draw.line((center_x, 0, center_x, height), fill="black", width=2)  # Y-Achse

        # Axis labels
        font = ImageFont.load_default()
        draw.text((width - 100, center_y + 10), "X-axis", fill="black", font=font)
        draw.text((center_x + 10, 10), "Y-axis", fill="black", font=font)
        draw.text((width // 4, 10), "Map created by the FastSLAM 2.0 algorithm", fill="black", font=font)

        return image, draw

    @staticmethod
    def __plot_as_arrows(draw: ImageDraw.Draw, directed_points: list[DirectedPoint], scale: float, color: str):
        """
        Plot the passed directed points as arrows with the passed scale and color.
        :param directed_points: This list contains all the directed points which will be represented as arrows
        :param scale: The scale of the arrow
        :param color: The color of the arrow
        """
        center_x = 750  # Middle of the X-axis
        center_y = 750  # Middle of the Y-axis
        for obj in directed_points:
            # Calculate the start and end point of the arrow
            x_start = center_x + obj.x * 50  # Scale the X-coordinate
            y_start = center_y - obj.y * 50  # Scale the Y-coordinate
            x_end = x_start + np.cos(obj.yaw) * scale
            y_end = y_start - np.sin(obj.yaw) * scale
            # Draw the arrow
            draw.line((x_start, y_start, x_end, y_end), fill=color, width=3)
            # Draw the arrow head
            arrow_size = 5
            draw.line((x_end, y_end, x_end - arrow_size * np.cos(obj.yaw + np.pi / 6),
                       y_end + arrow_size * np.sin(obj.yaw + np.pi / 6)), fill=color, width=3)
            draw.line((x_end, y_end, x_end - arrow_size * np.cos(obj.yaw - np.pi / 6),
                       y_end + arrow_size * np.sin(obj.yaw - np.pi / 6)), fill=color, width=3)

    @staticmethod
    def __plot_as_dots(draw: ImageDraw.Draw, points: list[Point], color: str):
        """
        Plot the passed points as dots. The color of the dots is determined by the passed color parameter.
        :param points: This list contains all the points which will be represented as dots in the map
        :param color: The color of the dot ('k' -> black, 'g' -> green)
        """
        for point in points:
            x = 750 + point.x * 50  # Scale the X-coordinate
            y = 750 - point.y * 50  # Scale the Y-coordinate
            radius = 3
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


class EvaluationService:
    @staticmethod
    def evaluate_estimation():
        # Get the actual position of the robot. Apply offset of start position (x=-1, y=1.5) so the robot starts at the origin (0, 0)
        actual_pos = Robot(
            HAL.getPose3d().x + 1,
            HAL.getPose3d().y - 1.5,
            HAL.getPose3d().yaw
        )

        # Calculate the deviation of the x coordinate in percentage
        x_deviation = EvaluationService.__calculate_linear_deviation(actual_pos.x, robot.x)

        # Calculate the deviation of the y coordinate in percentage
        y_deviation = EvaluationService.__calculate_linear_deviation(actual_pos.y, robot.y)

        # Calculate the deviation of the yaw angle in percentage
        angular_deviation = EvaluationService.__calculate_angular_deviation(actual_pos.yaw)

        # Calculate the average deviation of the robot in percentage
        average_deviation = (x_deviation + y_deviation + angular_deviation) / 3

        # Print the validation results
        print(f"\nAverage deviation: {average_deviation:.2f}%")
        print(f"X deviation: {x_deviation:.2f}%")
        print(f"Y deviation: {y_deviation:.2f}%")
        print(f"Angular deviation: {angular_deviation:.2f}%")

    @staticmethod
    def __calculate_linear_deviation(actual: float, estimated: float):
        """
        Calculate the linear deviation of the coordinate in percentage.
        :param actual: The actual coordinate of the robot
        :return: Returns the deviation of the x-coordinate in percentage
        """
        # Calculate the difference (delta) between the estimated and actual x-coordinates
        delta = actual - estimated

        # Avoid division by zero; if the estimated x is zero, return 0% deviation
        if estimated == 0:
            estimated = 0.001

        # Calculate the deviation percentage for the x-coordinate
        x_deviation_percentage = (abs(delta) / abs(estimated)) * 10  # Times 10 so 100% equals offset of 1 'meter'

        return x_deviation_percentage

    @staticmethod
    def __calculate_angular_deviation(actual_yaw: float) -> float:
        """
        Calculate the angular deviation of the robot in percentage.
        :param actual_yaw: The actual position of the robot
        :return: Returns the deviation of the yaw angle in percentage
        """
        # Calculate the angular deviation (absolute difference between yaw angles)
        angular_deviation = abs(actual_yaw - robot.yaw)

        # Normalize the angular deviation to be within the range [-pi, pi] (radians)
        angular_deviation = (angular_deviation + np.pi) % (2 * np.pi) - np.pi

        # Calculate the deviation percentage for the yaw angle
        return (abs(angular_deviation) / (2 * np.pi)) * 100  # Times 100 so 100% equals 2pi radians


# endregion

# region FastSLAM 2.0
class FastSLAM2:
    """
    Class that realizes the FastSLAM 2.0 algorithm.
    """

    def __init__(self):
        """
        Initialize the FastSLAM 2.0 algorithm with the specified number of particles.
        """
        # Initialize particles with the start position of the robot
        self.particles: list[Particle] = [
            Particle(
                x=0.0,
                y=0.0,
                yaw=0.0,
            ) for _ in range(NUM_PARTICLES)
        ]

    def iterate(self, d_linear: float, d_angular: float, measurements: list[Measurement]):
        """
        Perform one iteration of the FastSLAM 2.0 algorithm using the passed linear and angular delta values and the measurements.
        :param d_linear: linear delta value
        :param d_angular: angular delta value
        :param measurements: list of measurements to observed landmarks (distances and angles of landmark to robot and landmark ID)
        """
        # Update particle poses
        self.__move_particles(d_linear, d_angular)

        # Update particles (landmarks and weights)
        for measurement in measurements:
            for particle in self.particles:

                # Search for the associated landmark by the landmark ID of the measurement
                associated_landmark_index = particle.get_index_of_landmark(measurement.landmark_id)

                # If no associated landmark is found, the measurement is referencing to a new landmark
                # and the new landmark will be added to the particle map
                if associated_landmark_index is None:
                    landmark_x = particle.x + measurement.distance * math.cos(particle.yaw + measurement.yaw)
                    landmark_y = particle.y + measurement.distance * math.sin(particle.yaw + measurement.yaw)
                    particle.landmarks.append(Landmark(measurement.landmark_id, landmark_x, landmark_y))

                # If an associated landmark is found, the particle's map will be updated based on the actual measurement
                else:
                    # Get the associated landmark
                    associated_landmark = particle.landmarks[associated_landmark_index]

                    # Calculate the predicted measurement of the particle and the associated landmark
                    dx = associated_landmark.x - particle.x
                    dy = associated_landmark.y - particle.y
                    q = dx ** 2 + dy ** 2
                    distance = math.sqrt(q)
                    angle = np.arctan2(dy, dx) - particle.yaw
                    predicted_measurement = np.array([distance, angle])

                    # Calculate the innovation which is the difference between the actual measurement and the predicted measurement
                    innovation = measurement.as_vector() - predicted_measurement
                    innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi  # Ensure angle is between -pi and pi

                    # Calculate the Jacobian matrix of the particle and the associated landmark.
                    # Jacobian describes how changes in the state of the robot influence the measured observations.
                    # It helps to link the uncertainties in the estimates with the uncertainties in the measurements
                    jacobian = np.array([
                        [dx / distance, dy / distance],
                        [-dy / q, dx / q]
                    ])

                    # Calculate the covariance of the observation which depends on the Jacobian matrix,
                    # the covariance of the landmark and the measurement noise
                    observation_cov = jacobian @ associated_landmark.cov @ jacobian.T + MEASUREMENT_NOISE

                    # Calculate the Kalman gain which is used to update the pose/mean and covariance of the associated landmark.
                    # It determines how much the actual measurement should be trusted compared to the predicted measurement.
                    # Thus, it determines how much the landmark should be updated based on the actual measurement.
                    kalman_gain = associated_landmark.cov @ jacobian.T @ np.linalg.inv(observation_cov)

                    # Calculate updated pose/mean and covariance of the associated landmark
                    mean = associated_landmark.as_vector() + kalman_gain @ innovation
                    cov = (np.eye(len(associated_landmark.cov)) - kalman_gain @ jacobian) @ associated_landmark.cov

                    # Update the associated landmark
                    particle.landmarks[associated_landmark_index] = Landmark(
                        identifier=associated_landmark.id,
                        x=float(mean[0]),
                        y=float(mean[1]),
                        cov=cov
                    )

                    # Calculate the likelihood with the multivariate normal distribution
                    likelihood = multivariate_normal.pdf(innovation, mean=np.zeros(len(innovation)), cov=observation_cov)

                    # Update the particle weight with the likelihood
                    # print('\nlikelihood', likelihood)
                    particle.weight *= likelihood
                    # print('nw', particle.weight)

        # Normalisieren der Gewichte
        total_weight = sum(particle.weight for particle in self.particles)
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight

        weights = np.array([particle.weight for particle in self.particles])
        # print('\nNW', weights)

        # Überprüfen der Summe der Gewichte
        if np.sum(weights) == 0:
            raise ValueError("Die Summe der Partikelgewichte ist null! Überprüfen Sie die Gewichtungslogik.")

        # Effektive Partikelzahl
        N_eff = 1.0 / np.sum(weights ** 2)

        # Resampling nur durchführen, wenn N_eff klein ist
        if N_eff < len(self.particles) / 2:
            self.particles = self.systematic_resample()

    def __move_particles(self, d_linear: float, d_angular: float):
        """
        Update the poses of the particles based on the passed linear and angular delta values.
        :param d_linear: linear delta value
        :param d_angular: angular delta value
        """
        # Apply uncertainty to the movement of the robot and particles using random Gaussian noise with the standard deviations
        d_linear += random.gauss(0, TRANSLATION_NOISE)
        d_angular += random.gauss(0, ROTATION_NOISE)

        for p in self.particles:
            p.yaw = (p.yaw + d_angular + np.pi) % (2 * np.pi) - np.pi  # Ensure yaw stays between -pi and pi
            p.x += d_linear * np.cos(p.yaw)
            p.y += d_linear * np.sin(p.yaw)

    @staticmethod
    def __compute_jacobian(point_a: Point, point_b: Point):
        """
        Compute the Jacobian matrix of two points
        :param point_a: The first point
        :param point_b: The second point
        :return: The Jacobian matrix
        """
        dx = point_b.x - point_a.x
        dy = point_b.y - point_a.y
        q = dx ** 2 + dy ** 2
        distance = math.sqrt(q)

        return np.array([
            [dx / distance, dy / distance],
            [-dy / q, dx / q]
        ])

    def systematic_resample(self):
        num_particles = len(self.particles)

        # Normalisierung der Gewichte
        weights = np.array([particle.weight for particle in self.particles])
        weights /= np.sum(weights)  # Normalisierung der Gewichte

        # Resampling
        positions = (np.arange(num_particles) + np.random.rand()) / num_particles
        cumulative_sum = np.cumsum(weights)

        # Resampling mit Indices
        indices = np.searchsorted(cumulative_sum, positions)

        # Erstellen der neuen Partikel basierend auf den Indizes
        resampled_particles = [self.particles[i] for i in indices]

        return resampled_particles


# endregion

# region PARAMETERS
# Number of particles
NUM_PARTICLES = 30

# Distance threshold for associating landmarks to particles
MAXIMUM_LANDMARK_DISTANCE = 2

# Translation and rotation noise represent the standard deviation of the translation and rotation.
# The noise is used to add uncertainty to the movement of the robot and particles. It depends on the accuracy of the robot's odometry sensors.
TRANSLATION_NOISE = 0.01
ROTATION_NOISE = 0.001

# The measurement noise of the Kalman filter depends on the laser's accuracy
MEASUREMENT_NOISE = np.array([[0.01, 0.0], [0.0, 0.01]])
# endregion

# region FastSLAM 2.0 algorithm and objects in the environment
fast_slam = FastSLAM2()

# The robot that scans the environment and moves in the environment. It's position will be updated based on the particles of the FastSLAM 2.0 algorithm
robot = Robot()

# List of obstacles in the environment which will be plotted in the map. Only visualization purpose.
obstacles: list[Point] = []

# List of weighted/mean landmarks. The robot/particles will use these landmarks to estimate their position.
landmarks: list[Landmark] = []
# endregion

# The minimum number of iterations before updating the robot's position based on the estimated position of the particles
while True:
    # Move the robot and get the linear and angular delta values
    delta_linear, delta_angular = robot.move()

    # Get the points of scanned obstacles in the environment using the robot's laser data
    point_list = robot.scan_environment()

    # Search for landmarks in the scanned points using line filter and IEPF and get the measurements to them and their points
    measurement_list, landmark_points = LandmarkService.get_measurements_to_landmarks(point_list)

    # Update the landmark ID in the measurements if they are referencing to an existing landmark
    measurement_list = LandmarkService.associate_landmarks(measurement_list, landmark_points)

    # Iterate the FastSLAM 2.0 algorithm with the linear and angular velocities and the measurements to the observed landmarks
    fast_slam.iterate(delta_linear, delta_angular, measurement_list)

    # Update the robot's position based on the estimated position of the particles after a configured number of iterations
    (robot.x, robot.y, robot.yaw) = InterpretationService.estimate_robot_position(fast_slam.particles)

    # Plot the map with the robot, particles, landmarks and obstacles/borders
    MapService.plot_map()

    # Validate the robot's position based on the actual position
    EvaluationService.evaluate_estimation()
