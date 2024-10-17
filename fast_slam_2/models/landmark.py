import numpy as np
from numpy import ndarray

from fast_slam_2.models.point import Point


class Landmark(Point):
    """
    Class to represent a landmark in 2D space.
    A landmark has a covariance matrix which describes the uncertainty of the landmark's position.
    """

    def __init__(self, x: float, y: float, cov: ndarray = np.array([[0.1, 0], [0, 0.1]])):
        """
        Initialize the landmark with the passed parameters.
        :param x: The x coordinate of the landmark
        :param y: The y coordinate of the landmark
        :param cov: The covariance matrix of the landmark. Default is a 2x2 matrix with 0.1 on the diagonal.
        """
        super().__init__(x, y)
        self.cov = cov

    def __str__(self):
        """
        Get the string representation of the landmark.
        :return: Returns the string representation of the landmark.
        """
        return f"Landmark ID: x: {self.x}, y: {self.y}, Covariance: {self.cov}"
