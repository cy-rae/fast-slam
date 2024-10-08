import uuid

import numpy as np


class Measurement:
    """
    Class to represent the measurements of an observed landmark (distance and angle in radians).
    """

    def __init__(self, distance: float, yaw: float):
        self.landmark_id: uuid.UUID = uuid.uuid4()
        self.distance = distance
        self.yaw = yaw

    def as_vector(self):
        return np.array([self.distance, self.yaw])
