import random

import matplotlib.pyplot as plt
import numpy as np




class Landmark:
    """
    Class to represent a landmark
    """
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y

class Particle:
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

class Map:
    def plot_map(self, robot: Particle, particles: list[Particle], landmarks: list[Landmark]):
        """
        Plot the map with the robot, particles, and landmarks
        """
        self.__init_plot()
        self.__plot_robot(robot)
        self.__plot_particles(particles)
        self.__plot_landmarks(landmarks)

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
    def __plot_robot(robot: Particle):
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
    def __plot_particles(particles: list[Particle]):
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
    def __plot_landmarks(landmarks: list[Landmark]):
        """
        Plot the landmarks as black dots
        :param landmarks: This list contains all the landmark objects which contain the x and y values
        """
        for landmark in landmarks:
            plt.plot(landmark.x, landmark.y, 'ko')


class FastSlam:
    def __init__(self, num_particles: int):
        """
        Initialize the FastSLAM algorithm by initializing the robot, particles, landmarks, and map.
        :param num_particles: This is the number of particles to use in the FastSLAM algorithm
        """
        self.__robot = Particle()
        self.__particles: list[Particle] = [
            Particle(
                random.uniform(-4.1, 5.8),  # random x value
                random.uniform(-4.5, 5.5),  # random y value
                random.uniform(0, 360)  # random yaw value
            ) for _ in range(num_particles)
        ]
        self.__landmarks: list[Landmark] = []
        self.__map = Map()
        self.__map.plot_map(self.__robot, self.__particles, self.__landmarks)

    def iterate(self, v: float, w: float, laser_data: dict):
        """
        Iterate the FastSLAM algorithm
        :param v: This is the linear velocity of the robot
        :param w: This is the angular velocity of the robot
        :param laser_data: This is the laser data from the robot
        """
        self.__move(v, w)
        self.__update_landmarks(laser_data)


    def __move(self, v: float, w: float):
        """
        Move the robot and particles
        :param v: This is the linear velocity of the robot
        :param w: This is the angular velocity of the robot
        """
        self.__robot.move(v, w)
        for particle in self.__particles:
            particle.move(v, w)

    def __update_landmarks(self, laser_data: dict):
        pass

# Parameters
NUM_PARTICLES = 10

fast_slam = FastSlam(NUM_PARTICLES)