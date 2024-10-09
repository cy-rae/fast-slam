﻿import numpy as np


class Point:
    """
    Class to represent a point in 2D space.
    """

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def as_vector(self):
        """
        Get the pose/mean of the point as a vector [x, y].
        :return: Returns the position of the point as a numpy array [x, y]
        """
        return np.array([self.x, self.y])


    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
        }

    @staticmethod
    def from_dict(data):
        """
        Create a landmark object from the passed data
        :param data: This data contains the properties of the landmark
        :return: Returns the landmark object
        """
        return Point(
            x=data['x'],
            y=data['y']
        )