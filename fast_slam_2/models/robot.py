import math

import HAL
import numpy as np
from numpy import ndarray

from fast_slam_2.algorithms.icp import ICP
from fast_slam_2.config import TRANSLATION_NOISE, ROTATION_NOISE
from fast_slam_2.models.directed_point import DirectedPoint
from fast_slam_2.utils.evaluation_utils import EvaluationUtils


class Robot(DirectedPoint):
    """
    This class represents the robot
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        super().__init__(x, y, yaw)

        # Initialize the robot with the first scan
        self.__prev_points: ndarray = self.scan_environment()
        self.__prev_timestamp: int = HAL.getLaserData().timeStamp
        self.__prev_x = HAL.getPose3d().x
        self.__prev_y = HAL.getPose3d().y
        self.__prev_yaw = HAL.getPose3d().yaw

    @staticmethod
    def scan_environment() -> ndarray:
        """
        Scan the environment using the laser data and return a list of points that were scanned by the laser.
        :return: Returns a ndarray of scanned points which are also ndarray of x and y coordinates
        """
        # Get laser data from the robot. Laser data contains the distances and angles to obstacles in the environment.
        laser_data = HAL.getLaserData()

        # Convert each laser data value to a point
        scanned_points = []
        for i in range(180):  # Laser data has 180 values
            # Extract the distance at index i
            dist = laser_data.values[i]

            # If the distance is less than the minimum range or greater than the maximum range, skip the point
            if dist < laser_data.minRange or dist > laser_data.maxRange:
                continue

            # The final angle is centered (zeroed) at the front of the robot.
            angle = np.radians(i - 90)

            # Compute x, y coordinates from distance and angle
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)
            scanned_points.append([x, y])
        return np.array(scanned_points)

    @staticmethod
    def move(lin_velocity: float, ang_velocity: float) -> tuple[float, float]:
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
                w = ang_velocity
            else:  # left or center bumper
                w = -ang_velocity

        # If the robot does not hit the wall, the linear and angular velocities will be set to 1 and 0 respectively
        else:
            v = lin_velocity
            w = 0

        # Set the linear and angular velocity of the robot
        HAL.setV(v)
        HAL.setW(w)

        return v, w

    def icp_run(self, target_points: ndarray, v: float, w: float) -> tuple[float, float]:
        """
        Get the rotation and translation between the source and target points using the Iterative Closest Point (ICP) algorithm.
        :param target_points: Nx2 array of target points
        :param v: The linear velocity of the robot
        :param w: The angular velocity of the robot
        :return: Returns the rotation in radians and translation vector
        """
        # Set the current position to avoid false positive results due to time difference
        EvaluationUtils.set_actual_pos()

        # Get the rotation matrix and translation vector between the previous and target points using ICP
        rotation_matrix, translation_vector = ICP.run(self.__prev_points, target_points)

        # Update the previous points with the target points for the next iteration
        self.__prev_points = target_points

        # Covert the rotation matrix to an angle in radians
        rotation = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        # Compute the linear distance the robot has moved
        d_linear = np.linalg.norm(translation_vector)

        return rotation, d_linear

    def best_fit_transform(self, target_points: ndarray, v: float, w: float) -> tuple[float, float]:
        """
        Get the rotation and translation between the source and target points using the Iterative Closest Point (ICP) algorithm.
        :param target_points: Nx2 array of target points
        :param v: The linear velocity of the robot
        :param w: The angular velocity of the robot
        :return: Returns the rotation in radians and translation vector
        """
        # Set the current position to avoid false positive results due to time difference
        EvaluationUtils.set_actual_pos()

        # Get the rotation matrix and translation vector between the previous and target points using ICP
        rotation_matrix, translation_vector = ICP.best_fit_transform(self.__prev_points, target_points)

        # Update the previous points with the target points for the next iteration
        self.__prev_points = target_points

        # Covert the rotation matrix to an angle in radians
        rotation = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        # Compute the linear distance the robot has moved
        d_linear = np.linalg.norm(translation_vector)

        return rotation, d_linear

    def get_transformation(self, target_points: ndarray, v: float, w: float) -> tuple[float, float]:
        """
        Get the rotation and translation between the source and target points using the Iterative Closest Point (ICP) algorithm.
        :param target_points: Nx2 array of target points
        :param v: The linear velocity of the robot
        :param w: The angular velocity of the robot
        :return: Returns the rotation in radians and translation vector
        """
        # Set the current position to avoid false positive results due to time difference
        EvaluationUtils.set_actual_pos()

        # Get the rotation matrix and translation vector between the previous and target points using ICP
        rotation_matrix, translation_vector = ICP.get_transformation(self.__prev_points, target_points)

        # Update the previous points with the target points for the next iteration
        self.__prev_points = target_points

        # Covert the rotation matrix to an angle in radians
        rotation = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

        # Compute the linear distance the robot has moved
        d_linear = np.linalg.norm(translation_vector)

        return rotation, d_linear

    def get_displacement(self, v: int, w: int) -> tuple[float, float]:
        """
        Get the linear and angular displacement of the robot based on the linear and angular velocity.
        :param v: The linear velocity of the robot
        :param w: The angular velocity of the robot
        :return: Returns the linear and angular displacement of the robot as a tuple (d_lin, d_ang)
        """
        # Get the difference in time between the current and previous timestamp and update the previous timestamp
        curr_timestamp: int = HAL.getLaserData().timeStamp

        # Set the current position to avoid false positive results due to time difference
        EvaluationUtils.set_actual_pos()

        # Calculate the time difference between the current and previous timestamp
        dt: int = curr_timestamp - self.__prev_timestamp
        self.__prev_timestamp = curr_timestamp

        # Calculate the linear and angular displacement of the robot
        d_ang = w * dt
        d_lin = v * dt / 2

        return d_ang, d_lin