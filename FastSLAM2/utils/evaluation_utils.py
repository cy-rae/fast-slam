import HAL
import numpy as np

from FastSLAM2.models.directed_point import DirectedPoint


class EvaluationUtils:
    @staticmethod
    def evaluate_estimation(estimated_pos: DirectedPoint):
        """
        Evaluate the estimated position of the robot based on the actual position and print the deviation in percentage.
        :param estimated_pos: The estimated position of the robot
        """
        # Get the actual position of the robot. Apply offset of start position (x=-1, y=1.5) so the robot starts at the origin (0, 0)
        actual_pos = DirectedPoint(
            HAL.getPose3d().x + 1,
            HAL.getPose3d().y - 1.5,
            HAL.getPose3d().yaw
        )

        # Calculate the deviation of the x coordinate in percentage
        x_deviation = EvaluationUtils.__calculate_linear_deviation(actual_pos.x, estimated_pos.x)

        # Calculate the deviation of the y coordinate in percentage
        y_deviation = EvaluationUtils.__calculate_linear_deviation(actual_pos.y, estimated_pos.y)

        # Calculate the deviation of the yaw angle in percentage
        angular_deviation = EvaluationUtils.__calculate_angular_deviation(actual_pos.yaw, estimated_pos.yaw)

        # Calculate the average deviation of the robot in percentage
        average_deviation = (x_deviation + y_deviation + angular_deviation) / 3

        # Print the validation results
        print(f"\nAverage deviation: {average_deviation:.2f}%")
        print(f"X deviation: {x_deviation:.2f}%")
        print(f"Y deviation: {y_deviation:.2f}%")
        print(f"Angular deviation: {angular_deviation:.2f}%")

    @staticmethod
    def __calculate_linear_deviation(actual: float, estimated: float):
        """
        Calculate the linear deviation of the coordinate in percentage.
        :param actual: The actual coordinate of the robot
        :param estimated: The estimated coordinate of the robot
        :return: Returns the deviation of the x-coordinate in percentage
        """
        # Calculate the difference (delta) between the estimated and actual x-coordinates
        delta = actual - estimated

        # Avoid division by zero; if the estimated x is zero, return 0% deviation
        if estimated == 0:
            estimated = 0.001

        # Calculate the deviation percentage for the x-coordinate
        x_deviation_percentage = (abs(delta) / abs(estimated)) * 10  # Times 10 so 100% equals offset of 1 'meter'

        return x_deviation_percentage

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
        return (abs(angular_deviation) / (2 * np.pi)) * 100  # Times 100 so 100% equals 2pi radians
