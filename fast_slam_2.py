import math
import random

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.cluster import DBSCAN


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
    Class to represent a particle in the FastSLAM 2.0 algorithm.
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
            if distance < MAXIMUM_POINT_DISTANCE:
                return self.landmarks.index(particle_landmark)

        return None


class FastSLAM2:
    """
    Class that realizes the FastSLAM 2.0 algorithm.
    """

    def __init__(self):
        """
        Initialize the FastSLAM 2.0 algorithm with the specified number of particles.
        """
        self.particles: list[Particle] = [
            Particle(
                random.uniform(-4.1, 5.8),  # random x value
                random.uniform(-4.5, 5.5),  # random y value
                random.uniform(0, 360),  # random yaw value
            ) for _ in range(NUM_PARTICLES)
        ]

    def iterate(self, v: float, w: float, measurements: list[Measurement]):
        """
        Perform one iteration of the FastSLAM 2.0 algorithm using the passed linear and angular velocities and observations.
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
                    landmark_cov = np.array([[1e6, 0], [0, 1e6]])  # High uncertainty
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


# PARAMETERS
# Number of particles
NUM_PARTICLES = 100

# Distance-based clustering parameters
MAXIMUM_POINT_DISTANCE = 0.1
MIN_SAMPLES = 6

# Translation and rotation noise represent the standard deviation of the translation and rotation.
# The noise is used to add uncertainty to the movement of the robot and particles. It depends on the accuracy of the robot's odometry sensors.
TRANSLATION_NOISE = 0.1
ROTATION_NOISE = 0.1

# The measurement noise of the Kalman filter depends on the laser's accuracy
MEASUREMENT_NOISE = np.array([[0.1, 0.0], [0.0, 0.1]])


# -- JDE ROBOTS SPECIFIC -----------------------------------------------------------------------------------------------------------------------------------------

class LaserData:
    """
    Class to represent the laser data
    """

    def __init__(self, min_angle: float, max_angle: float, min_range: float, max_range: float, values: list[float]):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_range = min_range
        self.max_range = max_range
        self.values = values


def create_laser_data() -> LaserData:
    return LaserData()


class Robot(DirectedPoint):
    """
    This class represents the robot
    """

    def __init__(self):
        super().__init__(0, 0, 0)

    def scan_environment(self, laser_data: LaserData) -> list[Point]:
        """
        Scan the environment using the laser data and return a list of points that were scanned by the laser.
        :param laser_data: The laser data from the robot which contain the information about obstacles in the environment
        :return: Return a list of points that were scanned by the laser
        """
        # Get the angle increment, so we can calculate the angle of each laser beam.
        angle_increment = (laser_data.max_angle - laser_data.min_angle) / len(laser_data.values)

        # Iterate over the laser data values and calculate the point coordinates of the scanned obstacles
        scanned_points: list[Point] = []
        for i, distance in enumerate(laser_data.values):
            # If the distance is smaller than the minimum range or larger than the maximum range, no obstacle was found
            if distance < laser_data.min_range or distance > laser_data.max_range:
                continue

            # Calculate the angle of the laser beam in global coordinates. The angle is the sum of the robot's yaw and the laser angle.
            laser_angle = laser_data.min_angle + i * angle_increment
            global_angle = self.yaw + laser_angle

            # Calculate the point coordinates in global coordinates
            obstacle_x = distance * np.cos(np.radians(global_angle))
            obstacle_y = distance * np.sin(np.radians(global_angle))

            # Add the obstacle to the scanned obstacles list
            scanned_points.append(Point(obstacle_x, obstacle_y))

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
        observations: list[Measurement] = []
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
            observations.append(Measurement(distance, angle))

        return observations


def update_obstacles(scanned_obstacles: list[Point]):
    """
    Filter out the new obstacles which will be added to the obstacles list so the map will show new borders and obstacles.
    :param scanned_obstacles: The scanned obstacles
    """
    # Round the coordinates of the scanned obstacles to 2 decimal places to add noise to the data.
    # This is important since the laser data is not 100% accurate.
    # Thus, no new obstacles will be added to the same position when scanning the same obstacle multiple times.
    for scanned_obstacle in scanned_obstacles:
        scanned_obstacle.x = round(scanned_obstacle.x, 2)
        scanned_obstacle.y = round(scanned_obstacle.y, 2)

    # Update obstacles list with the scanned obstacles
    existing_coords = {(obstacle.x, obstacle.y) for obstacle in obstacles}
    new_obstacles = [obstacle for obstacle in scanned_obstacles if
                     (obstacle.x, obstacle.y) not in existing_coords]
    obstacles.extend(new_obstacles)


class Map:
    @staticmethod
    def plot_map(landmarks: list[Point]):
        """
        Plot the map with the robot, particles, landmarks and obstacles/borders.
        :param landmarks: The list of landmark objects
        """
        Map.__init_plot()
        Map.__plot_as_arrows(directed_points=[robot], scale=5.5, color='r')  # Plot the robot as a red arrow
        Map.__plot_as_arrows(directed_points=fast_slam.particles, scale=7,
                             color='b')  # Plot the particles as blue arrows
        Map.__plot_as_dots(obstacles, 'k')  # Mark obstacles as black dots
        Map.__plot_as_dots(landmarks, 'g')  # Mark landmarks as green dots

        # Save the plot as an image file and plot it
        plt.savefig('/usr/share/nginx/html/images/map.jpg')
        plt.show()

    @staticmethod
    def __init_plot():
        """
        Initialize the plot
        """
        plt.figure()
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Map created by the FastSLAM 2.0 algorithm')

    @staticmethod
    def __plot_as_arrows(directed_points: list[DirectedPoint], scale: float, color: str):
        """
        Plot the passed directed points as arrows with the passed scale and color.
        :param directed_points: This list contains all the directed points which will be represented as (short) arrows
        :param scale: The scale of the arrow
        :param color: The color of the arrow
        """
        plt.quiver(
            [obj.x for obj in directed_points],
            [particle.y for particle in directed_points],
            [np.cos(obj.get_yaw_rad()) for obj in directed_points],
            [np.sin(obj.get_yaw_rad()) for obj in directed_points],
            scale=scale,
            scale_units='inches',
            angles='xy',
            color=color
        )

    @staticmethod
    def __plot_as_dots(points: list[Point], color: str):
        """
        Plot the passed points as dots. The color of the dots is determined by the passed color parameter.
        :param points: This list contains all the points which will be represented as dots in the map
        :param color: The color of the dot ('k' -> black, 'g' -> green)
        """
        for landmark in points:
            plt.plot(landmark.x, landmark.y, color + 'o')  # 'o' -> circle marker


fast_slam = FastSLAM2()
robot = Robot()
obstacles: list[Point] = []  # List of obstacles in the environment which will be plotted in the map

laser_beams = create_laser_data()

while True:
    v, w = 1, 1

    # Get the points of scanned obstacles in the environment using the robot's laser data
    point_list = robot.scan_environment(laser_beams)

    # Update the obstacles list with the scanned points so new borders and obstacles will be added to the map
    update_obstacles(point_list)

    # Get the observations of the scanned landmarks
    measurement_list = robot.get_measurements_to_landmarks(point_list)

    # Iterate the FastSLAM 2.0 algorithm with the linear and angular velocities and the measurements to the observed landmarks
    fast_slam.iterate(v, w, measurement_list)

    # Todo: Get landmarks
    Map.plot_map([landmark for particle in fast_slam.particles for landmark in particle.landmarks])
