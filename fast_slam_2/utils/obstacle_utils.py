import numpy as np
from numpy import ndarray
from scipy.spatial import KDTree

from fast_slam_2.algorithms.line_filter import LineFilter
from fast_slam_2.models.directed_point import DirectedPoint
from fast_slam_2.models.point import Point


class ObstacleUtils:
    """
    This class is responsible for the management of all known obstacles.
    """
    obstacles: list[Point] = []
    threshold = 0.5

    @staticmethod
    def upsert_points(scanned_points: ndarray, robot: DirectedPoint):
        """
        Add or update the scanned points to the list of known obstacles.
        :param scanned_points: The scanned points from the robot's perspective as a Nx2 array.
        :param robot: The robot's current position and orientation.
        """
        # Filter the points to reduce noice
        filtered_points = LineFilter.filter(scanned_points)

        # Get the points from the robot's perspective
        observed_points: list[Point] = []
        for point in filtered_points:
            # Calculate the point in the robot's perspective
            x = robot.x + point[0] * np.cos(robot.yaw) - point[1] * np.sin(robot.yaw)
            y = robot.y + point[0] * np.sin(robot.yaw) + point[1] * np.cos(robot.yaw)
            observed_points.append(Point(x, y))

        if len(ObstacleUtils.obstacles) == 0:
            # Add all new points to the obstacles list
            ObstacleUtils.obstacles.extend(observed_points)
        else:
            # Associate the new points with the existing points and update them
            ObstacleUtils._update_existing_points(observed_points)

    @staticmethod
    def _update_existing_points(observed_points: list[Point]):
        """
        Associate the observed points with the existing points and update them.
        :param observed_points: The observed points (global coordinates).
        """
        # Get the X, Y values of the existing obstacles
        existing_obstacles = np.array([[point.x, point.y] for point in ObstacleUtils.obstacles])

        # Use KD Tree to find the nearest points
        tree = KDTree(existing_obstacles)

        # Iterate over all observed points
        for point in observed_points:
            # Find the nearest point
            dist, index = tree.query(point.as_vector())

            # If the distance is less than the threshold, update the point
            if dist < ObstacleUtils.threshold:
                existing_point = ObstacleUtils.obstacles[index].as_vector()
                observed_point = point.as_vector()
                updated_point = (existing_point + observed_point) / 2
                ObstacleUtils.obstacles[index] = Point(updated_point[0], updated_point[1])

            # If the distance is greater than the threshold, add the point to the obstacles list
            else:
                ObstacleUtils.obstacles.append(point)