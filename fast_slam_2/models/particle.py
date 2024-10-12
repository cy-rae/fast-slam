from fast_slam_2.config import NUM_PARTICLES
from fast_slam_2.models.directed_point import DirectedPoint
from fast_slam_2.models.landmark import Landmark


class Particle(DirectedPoint):
    """
    Class to represent a particle in the fast_slam_2 2.0 algorithm.
    """

    def __init__(self, x: float, y: float, yaw: float):
        super().__init__(x, y, yaw)
        self.weight = 1.0 / NUM_PARTICLES
        self.landmarks: list[Landmark] = []

    def __str__(self):
        return f"Particle: x: {self.x}, y: {self.y}, yaw: {self.yaw}, weight: {self.weight}, landmarks: {self.landmarks}"
