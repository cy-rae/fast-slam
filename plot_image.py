import random

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from sklearn.cluster import DBSCAN


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


class Point:
    """
    Class to represent a point on the map
    """

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y

    def position(self) -> ndarray:
        """
        Get the position of the point as a numpy array / vector
        :return: Returns the position of the point as a numpy array / vector
        """
        return np.array([self.x, self.y], dtype=float)


class Landmark(Point):
    """
    Class to represent a landmark
    """

    def __init__(self, x: float, y: float, covariance_matrix: ndarray = np.array([[1.0, 0.0], [0.0, 1.0]])):
        """
        Initialize the landmark with the passed x and y values.
        :param x: The x coordinate
        :param y: The y coordinate
        :param covariance_matrix: The covariance matrix of the landmark. Default value is a diagonal matrix with 1.0 (high uncertainty) on the diagonal
        """
        super().__init__(x, y)
        # The covariance matrix of the landmark represents the uncertainty of the landmark's position.
        self.covariance_matrix: ndarray = covariance_matrix


class Robot(Point):
    """
    Class to represent a robot
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
        """
        Initialize the particle with the passed x, y, and yaw values.
        :param x: The x value (default is 0.0)
        :param y: The y value (default is 0.0)
        :param yaw: The angle of the particle in degrees (default is 0.0). Should be a number between 0 and 360.
        """
        super().__init__(x, y)
        self.yaw = yaw

    def get_yaw_rad(self):
        """
        Get the yaw value in radians
        :return: Yaw value in radians
        """
        return np.radians(self.yaw)

    def move(self, v: float, w: float):
        """
        Move the particle
        :param v: The linear velocity
        :param w: The angular velocity
        """
        self.yaw = (self.yaw + w) % 360  # Ensure yaw stays between 0 and 360
        self.x += v * np.cos(self.get_yaw_rad())
        self.y += v * np.sin(self.get_yaw_rad())


class Particle(Robot):
    """
    Class to represent a weighted particle.
    """

    def __init__(self, x: float, y: float, yaw: float, weight: float):
        """
        Initialize the particle with the passed x, y, yaw, and weight values.
        :param x: The x coordinate
        :param y: The y coordinate
        :param yaw: The angle of the particle in degrees
        :param weight: The weight of the particle
        """
        super().__init__(x, y, yaw)
        self.weight: float = weight
        # A particle holds a list of landmarks that it has captured. This list represents the map of the particle.
        self.landmarks: list[Landmark] = []

    def update_landmarks(self, captured_landmarks: list[Landmark]):
        """
        Update the landmarks of the particle with the passed newly captured landmarks.
        :param captured_landmarks: The landmarks to update
        """
        # Iterate over the newly captured landmarks and update particle's landmarks list
        for captured_landmark in captured_landmarks:
            # Check if the newly captured landmark is associated to an existing landmark in the particle's landmarks list.
            associated_landmark_index: int or None = self.__get_associated_landmark(captured_landmark)

            # If no associated landmark was found, add the captured landmark to the particle's landmarks list.
            if associated_landmark_index is None:
                self.landmarks.append(captured_landmark)

            # If an associated landmark was found, update the landmark's position using Kalman update.
            else:
                self.__update_landmark(associated_landmark_index, captured_landmark)

    def __get_associated_landmark(self, captured_landmark: Landmark):
        """
        Search for an existing landmark in the particle's landmark list that is associated with the passed landmark
        using mahalanobis distance.
        :param captured_landmark: A landmark that is associated to this landmark will be searched.
        :return: Returns None if no landmark can be found. Else, the index of the associated landmark will be returned.
        """
        for particle_landmark in self.landmarks:
            # Calculate the mahalanobis distance between the captured landmark and the particle's landmark
            delta = captured_landmark.position() - particle_landmark.position()
            distance = np.sqrt(delta.T @ np.linalg.inv(particle_landmark.covariance_matrix) @ delta)

            # Use cluster radius as threshold for association
            if distance < MAXIMUM_POINT_DISTANCE:
                return self.landmarks.index(particle_landmark)

        return None

    def __update_landmark(self, particle_landmark_index: int, captured_landmark: Landmark):
        """
        Update the position of the particle's landmark using Kalman filter.
        :param particle_landmark_index: The index of the particle's landmark that will be updated
        :param captured_landmark: The captured landmark that will be used to update the particle's landmark
        """
        # Get the particle's landmark
        particle_landmark = self.landmarks[particle_landmark_index]

        # Calculate Kalman filter to enhance the landmark position estimate
        kalman_filter = particle_landmark.covariance_matrix @ np.linalg.inv(
            particle_landmark.covariance_matrix + MEASUREMENT_NOISE)

        # Update the landmark's position and covariance matrix using the Kalman filter
        updated_position = particle_landmark.position() + kalman_filter @ (
                captured_landmark.position() - particle_landmark.position())
        updated_covariance = (np.eye(2) - kalman_filter) @ particle_landmark.covariance_matrix

        # Update the particle's landmark with the updated position and covariance matrix
        updated_x, updated_y = updated_position[0], updated_position[1]
        self.landmarks[particle_landmark_index] = Landmark(float(updated_x), float(updated_y), updated_covariance)
        return updated_position, updated_covariance


class Map:
    def plot_map(
            self,
            robot: Robot,
            particles: list[Robot],
            obstacles: list[Point],
            landmarks: list[Point]
    ):
        """
        Plot the map with the robot, particles, and landmarks
        :param robot: The robot object
        :param particles: The list of particle objects
        :param obstacles: The list of obstacle objects
        :param landmarks: The list of landmark objects
        """
        self.__init_plot()
        self.__plot_as_arrows(obj_list=[robot], scale=5.5, color='r')  # Plot the robot as a red arrow
        self.__plot_as_arrows(obj_list=particles, scale=7, color='b')  # Plot the particles as blue arrows
        self.__plot_as_dots(obstacles, 'k')  # Mark obstacles as black dots
        self.__plot_as_dots(landmarks, 'g')  # Mark landmarks as green dots

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
        plt.title('Map created by the FastSLAM algorithm')

    @staticmethod
    def __plot_as_arrows(obj_list: list[Robot] or list[Particle], scale: float, color: str):
        """
        Plot the passed objects as arrows with the passed scale and color.
        :param obj_list: This list contains all the objects which contain the x, y, and yaw values
        :param scale: The scale of the arrow
        :param color: The color of the arrow
        """
        plt.quiver(
            [obj.x for obj in obj_list],
            [particle.y for particle in obj_list],
            [np.cos(obj.get_yaw_rad()) for obj in obj_list],
            [np.sin(obj.get_yaw_rad()) for obj in obj_list],
            scale=7,
            scale_units='inches',
            angles='xy',
            color='b'
        )

    @staticmethod
    def __plot_as_dots(obstacles: list[Point], color: str):
        """
        Plot the passed objects as dots. The color of the dots is determined by the passed color parameter.
        :param obstacles: This list contains all the landmark objects which contain the x and y values
        :param color: The color of the dot ('k' -> black, 'g' -> green)
        """
        for landmark in obstacles:
            plt.plot(landmark.x, landmark.y, color + 'o')  # 'o' -> circle marker


class FastSlam:
    def __init__(self, num_particles: int, weight_threshold: float):
        """
        Initialize the FastSLAM algorithm by initializing the robot, particles, landmarks, and map.
        :param num_particles: This is the number of particles to use in the FastSLAM algorithm
        Thus, the algorithm does not need to handle too many landmarks.
        """
        self.__robot = Robot()
        self.__particles: list[Particle] = [
            Particle(
                random.uniform(-4.1, 5.8),  # random x value
                random.uniform(-4.5, 5.5),  # random y value
                random.uniform(0, 360),  # random yaw value
                0.0  # default weight of the particle
            ) for _ in range(num_particles)
        ]
        self.__obstacles: list[Point] = []
        self.__landmarks: list[Point] = []
        self.__map = Map()
        self.__map.plot_map(self.__robot, self.__particles, self.__obstacles, self.__landmarks)
        self.__weight_threshold = weight_threshold

    def iterate(self, v: float, w: float, laser_data: LaserData):
        """
        Iterate the FastSLAM algorithm
        :param v: This is the linear velocity of the robot
        :param w: This is the angular velocity of the robot
        :param laser_data: This is the laser data from the robot
        """
        # Move the robot and particles
        self.__move(v, w)

        # Get the scanned obstacles from the laser data and update the obstacles and landmarks
        scanned_obstacles: list[Point] = self.__get_scanned_obstacles(laser_data)
        self.__update_obstacles(scanned_obstacles)
        self.__update_landmarks(scanned_obstacles)
        self.__update_particles()

    def __move(self, v: float, w: float):
        """
        Move the robot and particles
        :param v: This is the linear velocity of the robot
        :param w: This is the angular velocity of the robot
        """
        self.__robot.move(v, w)
        for particle in self.__particles:
            particle.move(v, w)

    def __get_scanned_obstacles(self, laser_data: LaserData) -> list[Point]:
        """
        Capture obstacles from the laser data
        :param laser_data: The laser data from the robot which contain the information about obstacles in the environment
        :return: Return a list of obstacles that were scanned by the laser
        """
        # Get the angle increment, so we can calculate the angle of each laser beam.
        angle_increment = (laser_data.max_angle - laser_data.min_angle) / len(laser_data.values)

        # Iterate over the laser data values and calculate the obstacle coordinates
        scanned_obstacles: list[Point] = []
        for i, distance in enumerate(laser_data.values):
            # If the distance is smaller than the minimum range or larger than the maximum range, no obstacle was found
            if distance < laser_data.min_range or distance > laser_data.max_range:
                continue

            # Calculate the angle of the laser beam in global coordinates. The angle is the sum of the robot's yaw and the laser angle.
            laser_angle = laser_data.min_angle + i * angle_increment
            global_angle = self.__robot.yaw + laser_angle

            # Calculate the obstacle coordinates in global coordinates
            obstacle_x = distance * np.cos(np.radians(global_angle))
            obstacle_y = distance * np.sin(np.radians(global_angle))

            # Create new obstacle object and with the rounded coordinate values. The coordinates will be rounded to 2
            # decimal places to add noise to the data. This is important since the laser data is not 100% accurate.
            # Thus, no new obstacles will be added to the same position when scanning the same obstacle multiple times.
            obstacle_x = round(obstacle_x, 2)
            obstacle_y = round(obstacle_y, 2)
            scanned_obstacles.append(Point(obstacle_x, obstacle_y))

        return scanned_obstacles

    def __update_obstacles(self, scanned_obstacles: list[Point]):
        """
        Filter out the new obstacles which will be added to the obstacles list so the map will show new borders and obstacles.
        :param scanned_obstacles: The scanned obstacles
        """
        # Update obstacles list with the scanned obstacles
        existing_coords = {(obstacle.x, obstacle.y) for obstacle in self.__obstacles}
        new_obstacles = [obstacle for obstacle in scanned_obstacles if
                         (obstacle.x, obstacle.y) not in existing_coords]
        self.__obstacles.extend(new_obstacles)

    def __update_landmarks(self, scanned_obstacles: list[Point]):
        """
        Update the landmarks of the particles with the scanned obstacles.
        :param scanned_obstacles: The scanned obstacles
        """
        # Extract landmarks from the scanned obstacles
        captured_landmarks = self.__capture_landmarks(scanned_obstacles)

        # Update the landmarks of the particles with the captured landmarks
        for particle in self.__particles:
            particle.update_landmarks(captured_landmarks)

    @staticmethod
    def __capture_landmarks(scanned_obstacles: list[Point]) -> list[Landmark]:
        """
        Capture landmarks from the scanned obstacles using distance-based clustering.
        :param scanned_obstacles: The scanned obstacles
        :return: Return a list of captured landmarks
        """
        # Get scanned obstacles as points
        x_coords = [obstacle.x for obstacle in scanned_obstacles]
        y_coords = [obstacle.y for obstacle in scanned_obstacles]
        points = np.column_stack((x_coords, y_coords))

        # Use distance-based clustering to extract clusters which represent landmarks
        db = DBSCAN(eps=MAXIMUM_POINT_DISTANCE, min_samples=MIN_SAMPLES).fit(points)

        # Get the unique labels (-1, 0, 1, 2, ...)  which represent the clusters.
        labels: ndarray = db.labels_
        unique_labels: set[int] = set(labels)

        # Iterate over the clusters and calculate the centroids which represent the landmark points
        extracted_landmarks: list[Landmark] = []
        for label in unique_labels:
            #  The label -1 represents noise (isolated points) and can be skipped
            if label == -1:
                continue

            # Get the points which belong to the current cluster
            cluster_points = points[labels == label]

            # Calculate the centroid of the cluster
            x = np.mean(cluster_points[:, 0])
            y = np.mean(cluster_points[:, 1])

            # Create a new landmark object and add it to the landmarks list
            extracted_landmarks.append(Landmark(float(x), float(y)))

        return extracted_landmarks

    def __update_particles(self):
        """
        Update the weights of the particles and remove the particles under the weight threshold.
        """
        for particle in self.__particles:
            # Update the weight of the particle
            particle.weight = self.__calculate_particle_weight(particle)

            # Remove particle if weight is below threshold
            if particle.weight < self.__weight_threshold:
                self.__particles.remove(particle)

    def __calculate_particle_weight(self, particle):
        pass


# Parameters
NUM_PARTICLES = 10
WEIGHT_THRESHOLD = 0.2

# Distance-based clustering parameters
MAXIMUM_POINT_DISTANCE = 0.1
MIN_SAMPLES = 6

# The measurement noise of the Kalman filter depends on the laser's accuracy
MEASUREMENT_NOISE = np.array([[0.1, 0.0], [0.0, 0.1]])

fast_slam = FastSlam(NUM_PARTICLES, WEIGHT_THRESHOLD)
