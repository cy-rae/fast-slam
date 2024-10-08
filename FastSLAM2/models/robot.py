import math

import HAL
import numpy as np
from numpy import ndarray

from FastSLAM2.algorithms.icp import ICP
from FastSLAM2.models.directed_point import DirectedPoint


class Robot(DirectedPoint):
    """
    This class represents the robot
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        super().__init__(x, y, yaw)

        # Initialize the robot with the first scan
        self.__prev_points: ndarray = self.scan_environment()

    @staticmethod
    def scan_environment() -> ndarray:
        """
        Scan the environment using the laser data and return a list of points that were scanned by the laser.
        :return: Returns a ndarray of scanned points which are also ndarray of x and y coordinates
        """
        # Get laser data from the robot. Laser data contains the distances and angles to obstacles in the environment.
        laser_data = HAL.getLaserData()

        # Convert each laser data value to a point
        scanned_points = np.empty(180, dtype=ndarray[float])
        for i in range(180):  # Laser data has 180 values
            # Extract the distance at index i
            dist = laser_data.values[i]

            # Skip invalid distances (e.g., min or max range)
            if dist < laser_data.minRange or dist > laser_data.maxRange:
                continue

            # The final angle is centered (zeroed) at the front of the robot.
            angle = np.radians(i - 90)

            # Compute x, y coordinates from distance and angle
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)
            scanned_points[i] = np.array([x, y])
        return scanned_points

    @staticmethod
    def move():
        """
        Set the linear and angular velocity of the robot based on the bumper state.
        :return: Returns the linear and angular velocity of the robot
        """
        # First, move robot in real world
        # Set linear and angular velocity depending on the bumper state.
        bumper_state = HAL.getBumperData().state
        if bumper_state == 1:
            # If the robot hits the wall, the linear velocity will be set to 0
            v = 0

            # If the robot hits the wall, the angular velocity will be set depending on the bumper that was hit
            bumper = HAL.getBumperData().bumper
            if bumper == 0:  # right bumper
                w = 1
            else:  # left or center bumper
                w = -1

        # If the robot does not hit the wall, the linear and angular velocities will be set to 1 and 0 respectively
        else:
            v = 1
            w = 0

        # Set the linear and angular velocity of the robot
        HAL.setV(v)
        HAL.setW(w)

    def get_transformation(self, target_points: ndarray) -> tuple[float, ndarray]:
        """
        Get the rotation and translation between the source and target points using the Iterative Closest Point (ICP) algorithm.
        :param target_points: Nx2 array of target points
        :return: Returns the rotation in radians and translation vector
        """
        # Get the rotation matrix and translation vector between the previous and target points using ICP
        rotation_matrix, translation_vector = ICP.run(self.__prev_points, target_points)

        # Covert the rotation matrix to an angle in radians
        rotation = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        # Update the previous points with the target points for the next iteration
        self.__prev_points = target_points

        return rotation, translation_vector
