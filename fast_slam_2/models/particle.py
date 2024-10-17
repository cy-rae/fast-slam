from fast_slam_2.config import NUM_PARTICLES
from fast_slam_2.models.directed_point import DirectedPoint
from fast_slam_2.models.landmark import Landmark


class Particle(DirectedPoint):
    """
    Class to represent a particle in the FastSLAM 2.0 algorithm.
    """

    def __init__(self, x: float, y: float, yaw: float):
        """
        Initialize the particle with the passed parameters.
        :param x: The x coordinate of the particle
        :param y: The y coordinate of the particle
        :param yaw: The angle of the particle in radians
        """
        super().__init__(x, y, yaw)
        self.weight = 1.0 / NUM_PARTICLES
        self.landmarks: list[Landmark] = []

    def __str__(self):
        """
        Get the string representation of the particle.
        :return: Returns the string representation of the particle.
        """
        return f"Particle: x: {self.x}, y: {self.y}, yaw: {self.yaw}, weight: {self.weight}, landmarks: {self.landmarks}"
