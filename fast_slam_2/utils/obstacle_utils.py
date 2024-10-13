import numpy as np
from numpy import ndarray

from fast_slam_2.algorithms.line_filter import LineFilter
from fast_slam_2.models.directed_point import DirectedPoint
from fast_slam_2.models.point import Point


class ObstacleUtils:
    """
    This class is responsible for the management of all known obstacles.
    """
    obstacles: list[Point]

    @staticmethod
    def update_obstacles(scanned_points: ndarray, robot: DirectedPoint):
        """
        Update the list of known obstacles with the scanned points.
        :param scanned_points: The scanned points as a numpy array
        """
        # Filter the points to reduce noice
        scanned_points = LineFilter.filter(scanned_points)

        # Get the points from the robot's perspective
        observed_points: list[Point] = []
        for point in scanned_points:
            # Calculate the point in the robot's perspective
            x = robot.x + point[0] * np.cos(robot.yaw) - point[1] * np.sin(robot.yaw)
            y = robot.y + point[0] * np.sin(robot.yaw) + point[1] * np.cos(robot.yaw)
            observed_points.append(Point(x, y))

        # Create a list to collect all new obstacles
        new_obstacles: list[Point] = []

        # Iterate through all scanned points
        for scanned_point in scanned_points:
            # Set a boolean that determines if the scanned point is a new obstacle
            is_new = True

            # Iterate through all known obstacles and check if the scanned point is already known
            for obstacle in ObstacleUtils.obstacles:
                # Calculate the euclidean distance between the scanned point and the known obstacle
                distance = np.sqrt(
                    (scanned_point[0] - obstacle.x) ** 2 + (scanned_point[1] - obstacle.y) ** 2
                )

                # If the distance is smaller than the threshold, the scanned point is not a new obstacle
                if distance <= 0.1:
                    is_new = False
                    break

            # If the scanned point is a new obstacle, add it to the list of new obstacles
            if is_new:
                new_obstacles.append(Point(scanned_point[0], scanned_point[1]))

        # Extend the list of known obstacles with the new obstacles
        ObstacleUtils.obstacles.extend(new_obstacles)