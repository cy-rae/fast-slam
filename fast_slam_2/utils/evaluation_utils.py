from datetime import datetime

import HAL
import numpy as np

from fast_slam_2.models.directed_point import DirectedPoint
from fast_slam_2.models.evaluation_results import EvaluationResults


class EvaluationUtils:
    """
    This class contains utility methods for evaluating the position estimation of the robot using the actual position data.
    """
    # Define the offset variables to move the actual robot position to the origin of the FastSLAM 2.0 map.
    initialized = False
    __offset_x: float
    __offset_y: float
    __offset_yaw: float

    # Variable for the actual robot position.
    __actual_pos: DirectedPoint

    @staticmethod
    def try_to_initialize():
        """
        Initialize the offset values for the actual robot position. The offset values are used to move the actual robot
        position to the correct position in the FastSLAM 2.0 map. The offset values are initialized if the offsets are
        not zero anymore. The simulation takes some time to start, so the initial robot position can be loaded after
        some iterations.
        """
        # Get the actual robot position
        x = HAL.getPose3d().x
        y = HAL.getPose3d().y
        yaw = HAL.getPose3d().yaw

        # Check if the actual robot position is not at the origin. If the actual robot position is at the origin, the
        # offset values are already initialized.
        if not np.isclose(x, 0.0, rtol=1e-09, atol=1e-09) or not np.isclose(y, 0.0, rtol=1e-09, atol=1e-09):
            # Initialize the offset values
            EvaluationUtils.__offset_x = x
            EvaluationUtils.__offset_y = y
            EvaluationUtils.__offset_yaw = yaw
            EvaluationUtils.initialized = True

    @staticmethod
    def set_actual_pos():
        """
        Set the actual position of the robot. The offset values are subtracted from the actual robot position to move the
        actual robot position to the origin of the FastSLAM 2.0 map.
        """
        EvaluationUtils.__actual_pos = DirectedPoint(
            HAL.getPose3d().x - EvaluationUtils.__offset_x,
            HAL.getPose3d().y - EvaluationUtils.__offset_y,
            HAL.getPose3d().yaw - EvaluationUtils.__offset_yaw
        )

    @staticmethod
    def evaluate_estimation(estimated_pos: DirectedPoint) -> tuple[EvaluationResults, DirectedPoint]:
        """
        Evaluate the estimated position of the robot based on the actual position and print the deviation in percentage.
        :param estimated_pos: The estimated position of the robot
        :return: Returns the evaluation results and the actual position of the robot
        """
        # Calculate the deviation of the x coordinate in percentage
        x_deviation, dx = EvaluationUtils.__calculate_linear_deviation(
            EvaluationUtils.__actual_pos.x,
            estimated_pos.x
        )

        # Calculate the deviation of the y coordinate in percentage
        y_deviation, dy = EvaluationUtils.__calculate_linear_deviation(
            EvaluationUtils.__actual_pos.y,
            estimated_pos.y
        )

        # Calculate the distance between the actual and estimated position
        distance = np.sqrt(dx ** 2 + dy ** 2)

        # Calculate the deviation of the yaw angle in percentage
        angular_deviation = EvaluationUtils.__calculate_angular_deviation(
            EvaluationUtils.__actual_pos.yaw,
            estimated_pos.yaw
        )

        # Calculate the average deviation of the robot in percentage
        average_deviation = (x_deviation + y_deviation + angular_deviation) / 3

        # Create results object
        timestamp: str = datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")
        results = EvaluationResults(
            timestamp,
            round(average_deviation, 2),
            round(x_deviation, 2),
            round(y_deviation, 2),
            round(angular_deviation, 2),
            round(distance, 4)
        )

        # Print the validation results
        print(f"\nTimestamp: {results.timestamp}")
        print(f"Average deviation: {results.average_deviation}%")
        print(f"X deviation: {results.x_deviation}%")
        print(f"Y deviation: {results.y_deviation}%")
        print(f"Angular deviation: {results.angular_deviation}%")
        print(f"Distance between actual and estimated position: {results.distance}m")

        return results, EvaluationUtils.__actual_pos

    @staticmethod
    def __calculate_linear_deviation(actual: float, estimated: float) -> tuple[float, float]:
        """
        Calculate the linear deviation of the coordinate in percentage.
        :param actual: The actual coordinate of the robot
        :param estimated: The estimated coordinate of the robot
        :return: Returns the deviation of the x-coordinate in percentage
        """
        # Calculate the difference (delta) between the estimated and actual x-coordinates
        delta = actual - estimated

        # Calculate the deviation percentage for the x-coordinate
        x_deviation_percentage = abs(delta) * 100  # Times 100 so 100% equals difference of 1 'meter'

        return x_deviation_percentage, delta

    @staticmethod
    def __calculate_angular_deviation(actual_yaw: float, estimated_yaw: float) -> float:
        """
        Calculate the angular deviation of the robot in percentage.
        :param actual_yaw: The actual angle of the robot
        :param estimated_yaw: The estimated angle of the robot
        :return: Returns the deviation of the angle in percentage
        """
        # Calculate the angular deviation (absolute difference between yaw angles)
        angular_deviation = abs(actual_yaw - estimated_yaw)

        # Normalize the angular deviation to be within the range [-pi, pi] (radians)
        angular_deviation = (angular_deviation + np.pi) % (2 * np.pi) - np.pi

        # Calculate the deviation percentage for the yaw angle
        return (abs(angular_deviation) / np.pi) * 100  # Times 100 so 100% equals pi radians (180 degrees)
