﻿"""
This script contains all the classes and methods that are needed to run the fast_slam_2 2.0 algorithm.
You can upload this script to the JDE Robots platform and run it in the simulation environment.
The script will create a map with the robot, particles, landmarks, and obstacles.
The robot will move in the environment and update its position based on the observed landmarks and the fast_slam_2 2.0 algorithm.
"""
import math
import random

import HAL
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from numpy import ndarray
from scipy import ndimage
from sklearn.cluster import DBSCAN, KMeans
import cv2


# region Models
class Point:
    """
    Class to represent a point in 2D space.
    """

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def pose(self):
        """
        Get the pose/mean of the point as a numpy array [x, y]
        :return:
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

    def get_yaw_rad(self):
        """
        Get the yaw value / current angle in radians
        :return: Yaw value in radians
        """
        return np.radians(self.yaw)


class Measurement:
    """
    Class to represent the measurements of an observed landmark (distance and angle in radians).
    """

    def __init__(self, distance: float, yaw: float):
        self.distance = distance
        self.yaw = yaw

    def as_vector(self):
        return np.array([self.distance, self.yaw])


class Landmark(Point):
    """
    Class to represent a landmark in 2D space.
    A landmark has a covariance matrix which describes the uncertainty of the landmark's position.
    """

    def __init__(self, x: float, y: float, cov: ndarray = np.array([[1.0, 0], [0, 1.0]])):
        super().__init__(x, y)
        self.cov = cov


class Particle(DirectedPoint):
    """
    Class to represent a particle in the fast_slam_2 2.0 algorithm.
    """

    def __init__(self, x: float, y: float, yaw: float):
        super().__init__(x, y, yaw)
        self.weight = 1.0 / NUM_PARTICLES
        self.landmarks: list[Landmark] = []

    def get_associated_landmark(self, measurement: Measurement) -> int or None:
        """
        Search for a landmark in the landmarks list that is associated with the observation that is described by the
        passed measurement using mahalanobis distance.
        :param measurement: The measurement of an observed landmark (distance and angle) to the robot.
        :return: Returns None if no landmark can be found. Else, the index of the associated landmark will be returned.
        """
        # Get the pose of the observed landmark starting from the particle
        observed_landmark_pose = np.array([
            self.x + measurement.distance * np.cos(self.get_yaw_rad() + measurement.yaw),
            self.y + measurement.distance * np.sin(self.get_yaw_rad() + measurement.yaw)
        ])

        for particle_landmark in self.landmarks:
            # Calculate the mahalanobis distance between the observed landmark and the particle's landmark
            # using the covariance matrix of the particle's landmark
            distance = Point.mahalanobis_distance(
                particle_landmark.pose(),
                observed_landmark_pose,
                particle_landmark.cov
            )

            # Use cluster radius as threshold for association
            if distance < MAXIMUM_LANDMARK_DISTANCE:
                return self.landmarks.index(particle_landmark)

        return None


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

    def get_measurements_to_landmarks(self, scanned_points: list[Point]) -> list[Measurement]:
        """
        Search for landmarks in passed list of points using distance-based clustering
        and measure their distances and angles to the robot. One cluster of points represents a landmark.
        :param scanned_points: The scanned obstacles as points.
        :return: The distances and angles from the observed landmarks to the robot will be returned.
        """
        # Get scanned obstacles/points as vectors
        x_coords = [obstacle.x for obstacle in scanned_points]
        y_coords = [obstacle.y for obstacle in scanned_points]
        points = np.column_stack((x_coords, y_coords))

        # Use distance-based clustering to extract clusters which represent landmarks
        db = DBSCAN(eps=MAXIMUM_POINT_DISTANCE, min_samples=MIN_SAMPLES).fit(points)

        # Get the unique labels (-1, 0, 1, 2, ...)  which represent the clusters.
        labels: ndarray = db.labels_
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


# endregion

# region Services
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
            MapService.__plot_as_arrows(draw, directed_points=[robot], scale=5.5, color='red')  # Plot the robot as a red arrow
            MapService.__plot_as_arrows(draw, directed_points=fast_slam.particles, scale=7, color='blue')  # Plot the particles as blue arrows
            MapService.__plot_as_dots(draw, obstacles, 'black')  # Mark obstacles as black dots
            MapService.__plot_as_dots(draw, landmarks, 'green')  # Mark landmarks as green dots

            # Save the plot as an image file
            image.save('/usr/shared/nginx/html/images/map.jpg', 'JPEG')
        except Exception as e:
            print(e)

    @staticmethod
    def __init_plot():
        """
        Initialize the plot
        """
        # Image size and background color
        width, height = 1000, 1000
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
        draw.text((width // 4, 10), "Map created by the fast_slam_2 2.0 algorithm", fill="black", font=font)

        return image, draw

    @staticmethod
    def __plot_as_arrows(draw: ImageDraw.Draw, directed_points: list[DirectedPoint], scale: float, color: str):
        """
        Plot the passed directed points as arrows with the passed scale and color.
        :param directed_points: This list contains all the directed points which will be represented as arrows
        :param scale: The scale of the arrow
        :param color: The color of the arrow
        """
        center_x = 500  # Middle of the X-axis
        center_y = 500  # Middle of the Y-axis
        for obj in directed_points:
            # Calculate the start and end point of the arrow
            x_start = center_x + obj.x * 50  # Skaliere die X-Koordinate
            y_start = center_y - obj.y * 50  # Skaliere die Y-Koordinate
            x_end = x_start + np.cos(obj.get_yaw_rad()) * scale
            y_end = y_start - np.sin(obj.get_yaw_rad()) * scale
            # Draw the arrow
            draw.line((x_start, y_start, x_end, y_end), fill=color, width=3)
            # Draw the arrow head
            arrow_size = 5
            draw.line((x_end, y_end, x_end - arrow_size * np.cos(obj.get_yaw_rad() + np.pi / 6),
                       y_end + arrow_size * np.sin(obj.get_yaw_rad() + np.pi / 6)), fill=color, width=3)
            draw.line((x_end, y_end, x_end - arrow_size * np.cos(obj.get_yaw_rad() - np.pi / 6),
                       y_end + arrow_size * np.sin(obj.get_yaw_rad() - np.pi / 6)), fill=color, width=3)

    @staticmethod
    def __plot_as_dots(draw: ImageDraw.Draw, points: list[Point], color: str):
        """
        Plot the passed points as dots. The color of the dots is determined by the passed color parameter.
        :param points: This list contains all the points which will be represented as dots in the map
        :param color: The color of the dot ('k' -> black, 'g' -> green)
        """
        for point in points:
            x = 500 + point.x * 50  # Scale the X-coordinate
            y = 500 - point.y * 50  # Scale the Y-coordinate
            radius = 3
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


class InterpretationService:
    """
    Service class to interpret the results of the fast_slam_2 2.0 algorithm.
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
                    scanned_obstacle.x * np.cos(robot.get_yaw_rad()) - scanned_obstacle.y * np.sin(robot.get_yaw_rad()))
            y_global = robot.y + (
                    scanned_obstacle.x * np.sin(robot.get_yaw_rad()) + scanned_obstacle.y * np.cos(robot.get_yaw_rad()))
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
    def get_weighted_landmarks(particles: list[Particle]) -> list[Landmark]:
        """
        Get the weighted landmarks by clustering the landmarks based on the particle weights using weighted k-means.
        :param particles: The weighted particles that contain a map with landmarks.
        :return: Returns a list with the weighted landmarks
        """
        # Get all landmarks and the corresponding particle weights
        landmark_poses_and_weights = [(landmark.pose(), particle.weight) for particle in particles for landmark in
                                      particle.landmarks]
        if len(landmark_poses_and_weights) == 0:
            return []
        (landmark_poses, weights) = zip(*landmark_poses_and_weights)

        # Get the number of clusters based on average number of landmarks per particle
        num_clusters = round(len(landmark_poses) / len(particles))

        # Use weighted k-means to cluster the landmarks based on the particle weights.
        # (The random state is set for reproducibility of random results which is useful for debugging)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(landmark_poses, sample_weight=weights)

        # Return the cluster centroids which represent the weighted landmarks
        centroids = kmeans.cluster_centers_
        return [Landmark(centroid[0], centroid[1]) for centroid in centroids]

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

        # Calculate the mean of the particles
        for p in particles:
            x_mean += p.x * p.weight
            y_mean += p.y * p.weight
            yaw_mean += p.yaw * p.weight

        # Normalize the estimated position
        x_mean /= total_weight
        y_mean /= total_weight
        yaw_mean /= total_weight

        return x_mean, y_mean, yaw_mean


# endregion

# region fast_slam_2 2.0
class FastSLAM2:
    """
    Class that realizes the fast_slam_2 2.0 algorithm.
    """

    def __init__(self):
        """
        Initialize the fast_slam_2 2.0 algorithm with the specified number of particles.
        """
        # self.particles: list[Particle] = [
        #     Particle(
        #         random.uniform(-4.1, 5.8),  # random x value
        #         random.uniform(-4.5, 5.5),  # random y value
        #         random.uniform(0, 360),  # random yaw value
        #     ) for _ in range(NUM_PARTICLES)
        # ]

        # Initialize particles with random values near the origin (0, 0, 0). A small variance will be used to add noise to the initial values.
        rng = np.random.default_rng(42)
        self.particles: list[Particle] = [
            Particle(
                x=rng.normal(0.0, 0.1),
                y=rng.normal(0.0, 0.1),
                yaw=rng.normal(0.0, 0.1),
            ) for _ in range(NUM_PARTICLES)
        ]

    def iterate(self, v: float, w: float, measurements: list[Measurement]):
        """
        Perform one iteration of the fast_slam_2 2.0 algorithm using the passed linear and angular velocities and observations.
        :param v: linear velocity
        :param w: angular velocity
        :param measurements: list of measurements to observed landmarks (distances and angles of landmark to robot)
        """
        # Update particle poses
        self.__move_particles(v, w)

        # Update particles (landmarks and weights)
        for measurement in measurements:
            for particle in self.particles:
                # Try to find an associated landmark for the observed landmark in the particle's landmarks list
                associated_landmark_index = particle.get_associated_landmark(measurement)

                if associated_landmark_index is None:
                    # If no associated landmark was found, add a new landmark to the particle's landmarks list
                    landmark_x = particle.x + measurement.distance * math.cos(particle.get_yaw_rad() + measurement.yaw)
                    landmark_y = particle.y + measurement.distance * math.sin(particle.get_yaw_rad() + measurement.yaw)
                    landmark_cov = np.array([[1, 0], [0, 1]])  # High uncertainty
                    particle.landmarks.append(Landmark(landmark_x, landmark_y, landmark_cov))

                else:
                    # If an associated landmark was found, update the landmark's position and covariance
                    associated_landmark = particle.landmarks[associated_landmark_index]

                    # Calculate the predicted measurement of the particle and the associated landmark
                    predicted_measurement = self.__get_predicted_measurement(particle, associated_landmark)

                    # Calculate the Jacobian matrix of the particle and the associated landmark.
                    # Jacobian describes how changes in the state of the robot influence the measured observations.
                    # It helps to link the uncertainties in the estimates with the uncertainties in the measurements
                    jacobian = self.__compute_jacobian(particle, associated_landmark)

                    # Calculate the covariance of the observation which depends on the Jacobian matrix,
                    # the covariance of the landmark and the measurement noise
                    observation_cov = jacobian @ associated_landmark.cov @ jacobian.T + MEASUREMENT_NOISE

                    # Calculate the Kalman gain which is used to update the pose/mean and covariance of the associated landmark.
                    # It determines how much the actual measurement should be trusted compared to the predicted measurement.
                    # Thus, it determines how much the landmark should be updated based on the actual measurement.
                    kalman_gain = associated_landmark.cov @ jacobian.T @ np.linalg.inv(observation_cov)

                    # Calculate the innovation which is the difference between the actual measurement and the predicted measurement
                    innovation = measurement.as_vector() - predicted_measurement

                    # Calculate updated pose/mean and covariance of the associated landmark
                    mean = associated_landmark.pose() + kalman_gain @ innovation
                    cov = (np.eye(2) - kalman_gain @ jacobian) @ associated_landmark.cov

                    # Update the associated landmark
                    particle.landmarks[associated_landmark_index] = Landmark(float(mean[0]), float(mean[1]), cov)

                    # Calculate the weight of the particle based on the likelihood of the observation
                    particle.weight *= self.__gaussian_likelihood(predicted_measurement, observation_cov,
                                                                  measurement.as_vector())

        # Resample particles
        self.__resample_particles()

    def __move_particles(self, v: float, w: float):
        """
        Update the poses of the particles based on the passed linear and angular velocities.
        :param v: linear velocity
        :param w: angular velocity
        """
        # Apply uncertainty to the movement of the robot and particles using random Gaussian noise with the standard deviations
        v += random.gauss(0, TRANSLATION_NOISE)
        w += random.gauss(0, ROTATION_NOISE)

        for p in self.particles:
            p.yaw = (p.yaw + w) % 360  # Ensure yaw stays between 0 and 360
            p.x += v * np.cos(p.get_yaw_rad())
            p.y += v * np.sin(p.get_yaw_rad())

    @staticmethod
    def __get_predicted_measurement(particle: Particle, landmark: Landmark):
        """
        Calculate the distance and angle from the passed particle to the passed landmark.
        :param particle: The particle from which the distance and angle to the landmark should be calculated.
        :param landmark: The landmark to which the distance and angle should be calculated.
        :return: The distance and angle from the particle to the landmark as a numpy array.
        """
        dx = landmark.x - particle.x
        dy = landmark.y - particle.y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        angle = math.atan2(dy, dx) - particle.get_yaw_rad()
        return np.array([distance, angle])

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
            [-dx / distance, -dy / distance],
            [dy / q, -dx / q]
        ])

    @staticmethod
    def __gaussian_likelihood(predicted_measurement: ndarray, cov: ndarray, actual_measurement: ndarray):
        """
        Calculate the Gaussian likelihood of the actual measurement given the predicted measurement and covariance.
        The likelihood is used to update the weight of the particle. It describes how likely the actual measurement fits the predicted measurement.
        :param predicted_measurement: The predicted measurement (distance and angle from the particle to the landmark)
        :param cov: Covariance matrix of the particle's landmark
        :param actual_measurement: The actual measurement (measured distance and angle to the landmark)
        :return: Returns the Gaussian likelihood of the actual measurement given the predicted measurement and covariance.
        """
        diff = actual_measurement - predicted_measurement
        exponent = -0.5 * np.dot(diff.T, np.linalg.inv(cov)).dot(diff)
        return math.exp(exponent) / math.sqrt((2 * math.pi) ** len(actual_measurement) * np.linalg.det(cov))

    def __resample_particles(self):
        """
        Resample the particles based on their weights. Particles with higher weights are more likely to be selected.
        Particles with lower weights are more likely to be removed. This helps to focus on the most likely particles.
        """
        particle_len = len(self.particles)

        # Normalize weights
        weights = np.array([p.weight for p in self.particles])
        normalized_weights = weights / np.sum(weights)

        # Create cumulative sum array
        cumulative_sum = np.cumsum(normalized_weights)

        # Resampling
        resampled_particles = []
        for _ in range(particle_len):
            # Get random number between 0 and 1
            r = random.random()

            # Add particle to resampled particles based on the cumulative sum
            for i in range(particle_len):
                if r < cumulative_sum[i]:
                    resampled_particles.append(self.particles[i])
                    break

        # Each particle gets the same weight after resampling
        for p in resampled_particles:
            p.weight = 1.0 / particle_len

        return resampled_particles


# endregion

def line_filter(points, sigma=1.0):
    """
    Glättet die Punkte mithilfe eines Gaußschen Filters.
    Args:
        points (ndarray): Ein Nx2 Array von [x, y] Punkten.
        sigma (float): Die Standardabweichung des Gauß-Filters.
    Returns:
        ndarray: Geglättete Punkte.
    """
    x_filtered = ndimage.gaussian_filter1d(points[:, 0], sigma=sigma)
    y_filtered = ndimage.gaussian_filter1d(points[:, 1], sigma=sigma)
    return np.vstack((x_filtered, y_filtered)).T

def hough_line_detection(image):
    # Schritt 1: Weichzeichnen des Bildes, um Rauschen zu reduzieren
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Schritt 2: Verwende die Canny-Kantendetektion
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    print('edges:', edges)

    # Schritt 3: Hough-Transformation zur Linienerkennung
    rho = 1                # Abstand-Auflösung in Pixel
    theta = np.pi / 180    # Winkelauflösung in Radianten
    threshold = 10         # Mindestanzahl der Stimmen, um eine Linie zu akzeptieren

    lines = cv2.HoughLines(edges, rho, theta, threshold)
    print('lines:', lines)

    return lines

def find_intersections(lines):
    # Finde die Schnittpunkte zwischen allen Linien
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            rho1, theta1 = lines[i][0]
            rho2, theta2 = lines[j][0]

            # Berechne die Schnittpunkte
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])

            # Lösen des linearen Gleichungssystems
            if np.linalg.det(A) != 0:  # Falls die Matrizen invertierbar sind
                intersection = np.linalg.solve(A, b)
                intersections.append(intersection.flatten())

    return np.array(intersections)
# Schritt 1: Linienfilter anwenden, um die Punktdaten zu glätten

def get_landmarks(scanned_points: np.ndarray):
    # Schritt 1: Finde den minimalen und maximalen Wert der Punkte, um das Bild korrekt zu erstellen
    min_x = int(np.min(scanned_points[:, 0]))
    min_y = int(np.min(scanned_points[:, 1]))
    max_x = int(np.max(scanned_points[:, 0]))
    max_y = int(np.max(scanned_points[:, 1]))

    # Verschiebung berechnen, um alle Punkte ins positive Koordinatensystem zu bringen
    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0

    # Neues Bild erstellen, welches alle Punkte enthält
    width = max_x + offset_x + 20  # zusätzliche Polsterung, um sicherzustellen, dass alle Punkte im Bild sind
    height = max_y + offset_y + 20
    image = np.zeros((height, width), dtype=np.uint8)

    # Schritt 2: Punkte in das Bild zeichnen (unter Berücksichtigung der Verschiebung)
    for point in scanned_points:
        x = int(point[0]) + offset_x
        y = int(point[1]) + offset_y
        cv2.circle(image, (x, y), 2, 255, -1)

    # Schritt 3: Hough-Transformation zur Linienerkennung
    lines = hough_line_detection(image)

    # Schritt 4: Finde die Schnittpunkte (Eckenerkennung)
    landmarks = []
    if lines is not None:
        intersections = find_intersections(lines)

        # Wandeln Sie Schnittpunkte in Landmark-Objekte um (unter Berücksichtigung der Verschiebung)
        for point in intersections:
            x = point[0] - offset_x
            y = point[1] - offset_y
            landmarks.append(Landmark(x=x, y=y))

    return landmarks

def calculate_distance_and_angle(point):
    """
    Calculate the distance and angle of a point to the origin (0, 0).
    :param point: Point(s) for which the distance and angle should be calculated
    :return: Returns the distance(s) and angle(s) of the point(s) to the origin (0, 0)
    """
    distance = np.sqrt(point[0] ** 2 + point[1] ** 2)
    angle = np.arctan2(point[1], point[0])
    return distance, angle


# region PARAMETERS
# Number of particles
NUM_PARTICLES = 60

# Distance threshold for associating landmarks to particles
MAXIMUM_LANDMARK_DISTANCE = 1

# Distance-based clustering parameters
MAXIMUM_POINT_DISTANCE = 0.1
MIN_SAMPLES = 10

# Translation and rotation noise represent the standard deviation of the translation and rotation.
# The noise is used to add uncertainty to the movement of the robot and particles. It depends on the accuracy of the robot's odometry sensors.
TRANSLATION_NOISE = 0.1
ROTATION_NOISE = 0.1

# The measurement noise of the Kalman filter depends on the laser's accuracy
MEASUREMENT_NOISE = np.array([[0.01, 0.0], [0.0, 0.01]])
# endregion

# region fast_slam_2 2.0 algorithm and objects in the environment
fast_slam = FastSLAM2()

# The robot that scans the environment and moves in the environment. It's position will be updated based on the particles of the fast_slam_2 2.0 algorithm
robot = Robot()

# List of obstacles in the environment which will be plotted in the map. Only visualization purpose.
obstacles: list[Point] = []

# List of weighted/mean landmarks. The robot/particles will use these landmarks to estimate their position.
landmarks: list[Landmark] = []
# endregion

# The minimum number of iterations before updating the robot's position based on the estimated position of the particles
MIN_ITERATIONS_TO_UPDATE_ROBOT_POSITION = 1000
iteration = 0
while True:
    # # Set linear velocity
    # v_i = 0.1
    #
    # # Set angular velocity. If the robot hits the wall, the angular velocity will be set to 0
    # bumper = HAL.getBumperData().state
    # if bumper == 1:
    #     w_i = 0.1
    # else:
    #     w_i = 0
    #
    # # Move robot
    # HAL.setV(v_i)
    # HAL.setW(w_i)

    if iteration < 100:
        HAL.setV(1.3)
        HAL.setW(-0.5)
        iteration += 1

    elif iteration < 140:
        HAL.setV(0)
        HAL.setW(0)
        iteration += 1


    else:
       # Get the points of scanned obstacles in the environment using the robot's laser data
       point_list = robot.scan_environment()

       point_data = np.array([point.pose() for point in point_list])
       filtered_points = line_filter(point_data)

       # Update the obstacles list with the scanned points so new borders and obstacles will be added to the map
       for point in filtered_points:
           obstacles.append(Landmark(point[0], point[1]))
       #obstacles = InterpretationService.update_obstacles(point_list)

       # Get the landmarks from the scanned points using hugh transformation
       landmarks = get_landmarks(filtered_points)

       # # Get the observations of the scanned landmarks
       # measurement_list = get_measurements_to_landmarks(point_list)
       #
       # # Iterate the fast_slam_2 2.0 algorithm with the linear and angular velocities and the measurements to the observed landmarks
       # fast_slam.iterate(v_i, w_i, measurement_list)
       #
       # # Update the robot's position based on the estimated position of the particles after a configured number of iterations
       # if iteration >= MIN_ITERATIONS_TO_UPDATE_ROBOT_POSITION and len(landmarks) > 3:
       #     (robot.x, robot.y, robot.yaw) = InterpretationService.estimate_robot_position(fast_slam.particles)
       # else:
       #     # Update the robot's position based on the current linear and angular velocities
       #     robot.x += v_i * np.cos(robot.get_yaw_rad())
       #     robot.y += v_i * np.sin(robot.get_yaw_rad())
       #     robot.yaw = (robot.yaw + w_i) % 360
       #
       # # Get the weighted landmarks by clustering the landmarks based on the particle weights
       # landmarks = InterpretationService.get_weighted_landmarks(fast_slam.particles)

       # Plot the map with the robot, particles, landmarks and obstacles/borders
       MapService.plot_map()

       # Increase iteration
       iteration += 1
