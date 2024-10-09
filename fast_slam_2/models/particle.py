import uuid

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

    def get_landmark(self, landmark_id: uuid.UUID) -> int or None:
        """
        Get the landmark and its index with the passed ID.
        :param landmark_id: The ID of the landmark
        :return: Returns the landmark  and its index or None if the landmark is not found
        """
        for i, landmark in enumerate(self.landmarks):
            if landmark.id == landmark_id:
                return landmark, i
        return None, None
