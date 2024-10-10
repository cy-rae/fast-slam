import uuid

import numpy as np
from numpy import ndarray

from fast_slam_2.models.point import Point


class Landmark(Point):
    """
    Class to represent a landmark in 2D space.
    A landmark has a covariance matrix which describes the uncertainty of the landmark's position.
    """

    def __init__(self, identifier: uuid.UUID, x: float, y: float, cov: ndarray = np.array([[0.1, 0], [0, 0.1]])):
        super().__init__(x, y)
        self.id = identifier
        self.cov = cov

    def __str__(self):
        return f"Landmark ID: {self.id}, x: {self.x}, y: {self.y}, Covariance: {self.cov}"