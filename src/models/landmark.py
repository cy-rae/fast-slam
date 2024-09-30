import numpy as np
from numpy import ndarray

from src.models.point import Point


class Landmark(Point):
    """
    Class to represent a landmark in 2D space.
    A landmark has a covariance matrix which describes the uncertainty of the landmark's position.
    """

    def __init__(self, x: float, y: float, cov: ndarray = np.array([[1.0, 0], [0, 1.0]])):
        super().__init__(x, y)
        self.cov = cov
