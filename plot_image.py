import random

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from sklearn.cluster import DBSCAN
import math


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


class Landmark:
    """
    Class to represent a landmark
    """

    def __init__(self, distance: float, yaw: float, covariance_matrix: ndarray = np.array([[1.0, 0.0], [0.0, 1.0]])):
        """
        Initialize the landmark with the passed distance and yaw values which represent the mean of the landmark.
        :param distance: The distance to the landmark
        :param yaw: The angle to the landmark
        :param covariance_matrix: The covariance matrix of the landmark. Default value is a diagonal matrix with 1.0 (high uncertainty) on the diagonal
        """
        self.distance = distance
        self.yaw = yaw
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
        self.yaw: float = yaw

    def get_yaw_rad(self):
        """
        Get the yaw value / current angle in radians
        :return: Yaw value in radians
        """
        return np.radians(self.yaw)

    def move(self, v: float, w: float):
        """
        Update the position using the passed linear and angular velocity. The passed values should already include noise.
        :param v: The linear velocity (including noise)
        :param w: The angular velocity (including noise)
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


class LandmarkService:
    """
    This class provides methods to update landmarks using mahalanobis distance and Kalman filter.
    """

    @staticmethod
    def update_landmarks(position: ndarray, existing_landmarks: list[Landmark], captured_landmarks: list[Landmark]) -> list[Landmark]:
        """
        Update the passed existing landmarks with the passed newly captured landmarks.
        :param position: The position of the particle / robot
        :param existing_landmarks: The landmarks to update
        :param captured_landmarks: The newly captured landmarks which will be used to update the existing landmarks
        :return: Returns a list the updated landmarks
        """
        # Iterate over the newly captured landmarks and update particle's landmarks list
        for captured_landmark in captured_landmarks:
            # Check if the newly captured landmark is associated to an existing landmark in the particle's landmarks list.
            associated_landmark_index: int or None = LandmarkService.__get_associated_landmark(existing_landmarks,
                                                                                               captured_landmark)

            # If no associated landmark was found, add the captured landmark to the particle's landmarks list.
            if associated_landmark_index is None:
                existing_landmarks.append(captured_landmark)

            # If an associated landmark was found, update the landmark's position using Kalman update.
            else:
                LandmarkService.__update_landmark(existing_landmarks, associated_landmark_index, captured_landmark)

        return existing_landmarks

    @staticmethod
    def __get_associated_landmark(existing_landmarks: list[Landmark], captured_landmark: Landmark):
        """
        Search for a landmark in the existing landmarks list that is associated with the passed captured landmark
        using mahalanobis distance.
        :param existing_landmarks: The existing landmarks list
        :param captured_landmark: A landmark that is associated to this landmark will be searched.
        :return: Returns None if no landmark can be found. Else, the index of the associated landmark will be returned.
        """
        for particle_landmark in existing_landmarks:
            # Calculate the mahalanobis distance between the captured landmark and the particle's landmark
            # using the landmark covariance matrix of the particle's landmark
            distance = LandmarkService.mahalanobis_distance(
                particle_landmark.position(),
                captured_landmark.position(),
                particle_landmark.covariance_matrix
            )

            # Use cluster radius as threshold for association
            if distance < MAXIMUM_POINT_DISTANCE:
                return existing_landmarks.index(particle_landmark)

        return None

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

    @staticmethod
    def __update_landmark(existing_landmarks: list[Landmark], particle_landmark_index: int,
                          captured_landmark: Landmark):
        """
        Update the position of the particle's landmark using Kalman filter.
        :param particle_landmark_index: The index of the particle's landmark that will be updated
        :param captured_landmark: The captured landmark that will be used to update the particle's landmark
        """
        # Get the particle's landmark
        particle_landmark = existing_landmarks[particle_landmark_index]

        # Calculate Kalman filter to enhance the landmark position estimate
        kalman_filter = particle_landmark.covariance_matrix @ np.linalg.inv(
            particle_landmark.covariance_matrix + MEASUREMENT_NOISE)

        # Update the landmark's position and covariance matrix using the Kalman filter
        updated_position = particle_landmark.position() + kalman_filter @ (
                captured_landmark.position() - particle_landmark.position())
        updated_covariance = (np.eye(2) - kalman_filter) @ particle_landmark.covariance_matrix

        # Update the particle's landmark with the updated position and covariance matrix
        updated_x, updated_y = updated_position[0], updated_position[1]
        existing_landmarks[particle_landmark_index] = Landmark(float(updated_x), float(updated_y), updated_covariance)
        return updated_position, updated_covariance


class FastSlam:
    def __init__(self):
        """
        Initialize the FastSLAM algorithm by initializing the robot, particles, landmarks, and map.
        """
        self.__robot = Robot()
        self.__particles: list[Particle] = [
            Particle(
                random.uniform(-4.1, 5.8),  # random x value
                random.uniform(-4.5, 5.5),  # random y value
                random.uniform(0, 360),  # random yaw value
                0.0  # default weight of the particle
            ) for _ in range(NUM_PARTICLES)
        ]
        self.__obstacles: list[Point] = []
        self.__landmarks: list[Landmark] = []
        self.__map = Map()
        self.__map.plot_map(self.__robot, self.__particles, self.__obstacles, self.__landmarks)

    def iterate(self, v: float, w: float, laser_data: LaserData):
        """
        Iterate the FastSLAM algorithm
        :param v: This is the linear velocity of the robot
        :param w: This is the angular velocity of the robot
        :param laser_data: This is the laser data from the robot
        """
        # Move the robot and particles
        self.__move_particles(v, w)

        # Get the scanned obstacles from the laser data and update the obstacles to illustrate new borders and obstacles in the map
        scanned_obstacles: list[Point] = self.__get_scanned_obstacles(laser_data)
        self.__update_obstacles(scanned_obstacles)

        # Capture landmarks from the scanned obstacles
        captured_landmarks = self.__capture_landmarks(scanned_obstacles)

        # Update the landmarks of the particles with the captured landmarks
        for particle in self.__particles:
            particle.landmarks = LandmarkService.update_landmarks(particle.position(), particle.landmarks, captured_landmarks)

        # Update the actual measured landmarks
        self.__landmarks = LandmarkService.update_landmarks(self.__landmarks, captured_landmarks)

        # Update the particles with regard to their weights. The weight depends on the likelihood of the particle given the captured landmarks.
        if len(captured_landmarks) > 0:
            self.__update_particles(captured_landmarks)

    def __move_particles(self, v: float, w: float):
        """
        Apply the movement to the robot to the particles. The uncertainty of the odometry measurements are taken into account.
        :param v: This is the linear velocity of the robot
        :param w: This is the angular velocity of the robot
        """
        # Apply uncertainty to the movement of the robot and particles using random Gaussian noise with the standard deviations
        v += random.gauss(0, TRANSLATION_NOISE)
        w += random.gauss(0, ROTATION_NOISE)

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

            # Add the obstacle to the scanned obstacles list
            scanned_obstacles.append(Point(obstacle_x, obstacle_y))

        return scanned_obstacles

    def __update_obstacles(self, scanned_obstacles: list[Point]):
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
        existing_coords = {(obstacle.x, obstacle.y) for obstacle in self.__obstacles}
        new_obstacles = [obstacle for obstacle in scanned_obstacles if
                         (obstacle.x, obstacle.y) not in existing_coords]
        self.__obstacles.extend(new_obstacles)

    def __capture_landmarks(self, scanned_obstacles: list[Point]) -> list[Landmark]:
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

            # Calculate the distance and angle of the landmark to the current robot position
            dx = x - self.__robot.x
            dy = y - self.__robot.y
            distance = math.sqrt(dx ** 2 + dy ** 2)
            angle = math.atan2(dy, dx) # Todo: Maybe take robot's yaw into account

            # Create a new landmark object and add it to the landmarks list
            extracted_landmarks.append(Landmark(distance, angle))

        return extracted_landmarks

    def __update_particles(self, captured_landmarks: list[Landmark]):
        """
        Update the weights of the particles based on the likelihood of the particle given the captured landmarks.
        """
        for particle in self.__particles:
            weight = 1.0

            for captured_landmark in captured_landmarks:
                # Estimated landmark position based on the particle's position
                dx = captured_landmark.x - particle.x
                dy = captured_landmark.y - particle.y
                predicted_range = np.sqrt(dx ** 2 + dy ** 2)
                predicted_bearing = np.arctan2(dy, dx) - particle.get_yaw_rad()

                # Actual measured landmark position
                measurement = np.array([captured_landmark.x, captured_landmark.y])
                mu = np.array([predicted_range, predicted_bearing])
                covariance = captured_landmark.covariance_matrix

                # Berechnung der Mahalanobis-Distanz
                distance = mahalanobis_distance(mu, measurement, covariance)

                # Calculate the mahalanobis distance between the captured landmark and the particle's landmark
                # using the landmark covariance matrix of the particle's landmark
                distance = LandmarkService.mahalanobis_distance(
                    particle_landmark.position(),
                    captured_landmark.position(),
                    particle_landmark.covariance_matrix
                )

                # Berechnung der Gewichtung basierend auf der Mahalanobis-Distanz
                # Du kannst eine Threshold oder eine Gewichtsfunktion verwenden
                if distance < 1.0:  # Beispielschwellenwert
                    likelihood = np.exp(-0.5 * distance ** 2)  # Gaußsche Verteilung
                    weight *= likelihood

            # Aktualisiere das Gewicht des Partikels
            particle['weight'] = weight
            updated_particles.append(particle)

    def __calculate_particle_weight(self, particle):
        pass


# Parameters
NUM_PARTICLES = 10
WEIGHT_THRESHOLD = 0.2

# Distance-based clustering parameters
MAXIMUM_POINT_DISTANCE = 0.1
MIN_SAMPLES = 6

# Translation and rotation noise represent the standard deviation of the translation and rotation.
# The noise is used to add uncertainty to the movement of the robot and particles. It depends on the accuracy of the robot's odometry sensors.
TRANSLATION_NOISE = 0.1
ROTATION_NOISE = 0.1

# The measurement noise of the Kalman filter depends on the laser's accuracy
MEASUREMENT_NOISE = np.array([[0.1, 0.0], [0.0, 0.1]])

fast_slam = FastSlam()
