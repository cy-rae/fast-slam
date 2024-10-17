import numpy as np


class Measurement:
    """
    Class to represent the measurements of an observed landmark (distance and angle in radians).
    """

    def __init__(self, distance: float, yaw: float):
        """
        Initialize the measurement with the passed parameters.
        :param distance: The distance to the landmark
        :param yaw: The angle to the landmark in radians
        """
        self.distance = distance
        self.yaw = yaw

    def as_vector(self):
        """
        Get the measurement as a vector [distance, yaw].
        :return: Returns the measurement as a Nx2 array [distance, yaw]
        """
        return np.array([self.distance, self.yaw])
