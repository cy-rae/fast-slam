import numpy as np


class Point:
    """
    Class to represent a point in 2D space.
    """

    def __init__(self, x: float, y: float):
        """
        Initialize the point with the passed variables.
        :param x: The x coordinate of the point
        :param y: The y coordinate of the point
        """
        self.x = x
        self.y = y

    def as_vector(self):
        """
        Get the point as a vector [x, y].
        :return: Returns the point as a Nx2 array [x, y]
        """
        return np.array([self.x, self.y])

    def to_dict(self):
        """
        Convert the point to a dictionary.
        :return: Returns the dictionary representation of the point.
        """
        return {
            'x': self.x,
            'y': self.y,
        }
