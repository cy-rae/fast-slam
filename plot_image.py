import random

import matplotlib.pyplot as plt
import numpy as np


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


class Obstacle:
    """
    Class to represent an obstacle or landmark
    """
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y


class Robot:
    """
    Class to represent a particle or robot
    """
    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
        """
        Initialize the particle with the passed x, y, and yaw values.
        :param x: The x value (default is 0.0)
        :param y: The y value (default is 0.0)
        :param yaw: The angle of the particle in degrees (default is 0.0). Should be a number between 0 and 360.
        """
        self.x = x
        self.y = y
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
        self.weight = weight

class Map:
    def plot_map(
            self,
            robot: Robot,
            particles: list[Robot],
            obstacles: list[Obstacle],
            landmarks: list[Obstacle]
    ):
        """
        Plot the map with the robot, particles, and landmarks
        :param robot: The robot object
        :param particles: The list of particle objects
        :param obstacles: The list of obstacle objects
        :param landmarks: The list of landmark objects
        """
        self.__init_plot()
        self.__plot_robot(robot)
        self.__plot_particles(particles)
        self.__plot_obstacles(obstacles)
        self.__plot_obstacles(landmarks, 'go')  # Mark landmarks as green dots

        # Save the plot as an image file and plot it
        plt.savefig('plot_image.jpg')
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
    def __plot_robot(robot: Robot):
        """
        Plot the robot as a red arrow
        :param robot: This robot object contains the x, y, and yaw values of the robot
        """
        plt.quiver(
            robot.x,
            robot.y,
            np.cos(robot.get_yaw_rad()),
            np.sin(robot.get_yaw_rad()),
            scale=5.5,
            scale_units='inches',
            angles='xy',
            color='r'
        )

    @staticmethod
    def __plot_particles(particles: list[Robot]):
        """
        Plot the particles as blue arrows
        :param particles: This list contains all the particle objects which contain the x, y, and yaw values
        """
        plt.quiver(
            [particle.x for particle in particles],
            [particle.y for particle in particles],
            [np.cos(particle.get_yaw_rad()) for particle in particles],
            [np.sin(particle.get_yaw_rad()) for particle in particles],
            scale=7,
            scale_units='inches',
            angles='xy',
            color='b'
        )

    @staticmethod
    def __plot_obstacles(obstacles: list[Obstacle], color: str = 'ko'):
        """
        Plot the landmarks as dots. The color of the dots is determined by the passed color parameter. If it is not set,
        the dots will be black.
        :param obstacles: This list contains all the landmark objects which contain the x and y values
        :param color: The color of the dot (default: 'ko' -> black dot)
        """
        for landmark in obstacles:
            plt.plot(landmark.x, landmark.y, color)


class FastSlam:
    def __init__(self, num_particles: int, landmark_limit: int, weight_threshold: float):
        """
        Initialize the FastSLAM algorithm by initializing the robot, particles, landmarks, and map.
        :param num_particles: This is the number of particles to use in the FastSLAM algorithm
        :param landmark_limit: This parameter determines how many landmarks should be stored in the landmarks list.
        Thus, the algorithm does not need to handle too many landmarks.
        """
        self.__robot = Robot()
        self.__particles: list[Particle] = [
            Particle(
                random.uniform(-4.1, 5.8),  # random x value
                random.uniform(-4.5, 5.5),  # random y value
                random.uniform(0, 360), # random yaw value
                0.0 # default weight of the particle
            ) for _ in range(num_particles)
        ]
        self.__obstacles: list[Obstacle] = []
        self.__landmarks: list[Obstacle] = []
        self.__landmark_limit = landmark_limit
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
        self.__move(v, w)
        self.__update_obstacles_and_landmarks(laser_data)
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

    def __update_obstacles_and_landmarks(self, laser_data: LaserData):
        """
        Capture new landmarks from the laser data and add them to the landmarks list.
        :param laser_data: The laser data from the robot
        """
        # Get the scanned obstacles from the laser data and filter out the new obstacles.
        scanned_obstacles = self.__get_scanned_obstacles(laser_data)
        new_obstacles = self.filter_new_obstacles(scanned_obstacles)

        # Update the list of obstacles and landmarks. Always add only one element of the new obstacles that is in the
        # middle of the list. Thus, the algorithm does not need to handle too many landmarks.
        self.__obstacles.extend(new_obstacles)
        if len(self.__landmarks) < self.__landmark_limit:
            middle_index = len(new_obstacles) // 2
            self.__landmarks.append(new_obstacles[middle_index])

    def __get_scanned_obstacles(self, laser_data: LaserData):
        """
        Capture obstacles from the laser data
        :param laser_data: The laser data from the robot which contain the information about obstacles in the environment
        :return: Return a list of obstacles that were scanned by the laser
        """
        # Get the angle increment, so we can calculate the angle of each laser beam.
        angle_increment = (laser_data.max_angle - laser_data.min_angle) / len(laser_data.values)

        # Iterate over the laser data values and calculate the obstacle coordinates
        scanned_obstacles: list[Obstacle] = []
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
            scanned_obstacles.append(Obstacle(obstacle_x, obstacle_y))

        return scanned_obstacles

    def filter_new_obstacles(self, scanned_obstacles: list[Obstacle]):
        """
        Filter out obstacles from the passed list that already exist in obstacles property list.
        :param scanned_obstacles: List of newly scanned obstacles
        :return: List of new obstacles that are not in the existing obstacles list
        """
        existing_coords = {(obstacle.x, obstacle.y) for obstacle in self.__obstacles}
        new_obstacles = [obstacle for obstacle in scanned_obstacles if
                         (obstacle.x, obstacle.y) not in existing_coords]
        return new_obstacles

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


# Parameters
NUM_PARTICLES = 10
LANDMARK_LIMIT = 100
WEIGHT_THRESHOLD = 0.2

fast_slam = FastSlam(NUM_PARTICLES, LANDMARK_LIMIT, WEIGHT_THRESHOLD)
