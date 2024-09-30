import math
import random


class LaserData:
    """
    Class to represent the laser data
    """

    def __init__(self, min_angle: float, max_angle: float, min_range: float, max_range: float, values: list[float]):
        self.minAngle = min_angle
        self.maxAngle = max_angle
        self.minRange = min_range
        self.maxRange = max_range
        self.values = values


class HAL:
    @staticmethod
    def getLaserData() -> LaserData:
        # Create test data with more values
        min_angle = 0  # Minimum angle in radians (-90 degrees)
        max_angle = math.pi  # Maximum angle in radians (90 degrees)
        min_range = 0.1  # Minimum range in meters
        max_range = 10.0  # Maximum range in meters
        values = []  # More example distance values
        for _ in range(180):
            values.append(random.uniform(1, 10))

        # Instantiate the class with the test data
        return LaserData(min_angle, max_angle, min_range, max_range, values)
