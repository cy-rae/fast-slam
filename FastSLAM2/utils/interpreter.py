import numpy as np

from FastSLAM2.models.particle import Particle
from FastSLAM2.models.point import Point


class Interpreter:
    """
    Service class to interpret the results of the FastSLAM2 2.0 algorithm.
    """

    @staticmethod
    def update_obstacles(scanned_obstacles: list[Point]):
        """
        Filter out the new obstacles which will be added to the obstacles list so the map will show new borders and obstacles.
        :param scanned_obstacles: The scanned obstacles
        """
        # Apply translation and rotation of the robot to the scanned obstacles to get the global coordinates
        global_points: list[Point] = []
        for scanned_obstacle in scanned_obstacles:
            x_global = robot.x + (
                    scanned_obstacle.x * np.cos(robot.yaw) - scanned_obstacle.y * np.sin(robot.yaw))
            y_global = robot.y + (
                    scanned_obstacle.x * np.sin(robot.yaw) + scanned_obstacle.y * np.cos(robot.yaw))
            global_points.append(Point(x_global, y_global))

        # Round the coordinates of the scanned obstacles to 2 decimal places to add noise to the data.
        # This is important since the laser data is not 100% accurate.
        # Thus, no new obstacles will be added to the same position when scanning the same obstacle multiple times.
        for global_point in global_points:
            global_point.x = round(global_point.x, 1)
            global_point.y = round(global_point.y, 1)

        # Update obstacles list with the scanned obstacles
        existing_coords = {(obstacle.x, obstacle.y) for obstacle in obstacles}
        new_obstacles = [obstacle for obstacle in global_points if
                         (obstacle.x, obstacle.y) not in existing_coords]
        obstacles.extend(new_obstacles)

        return obstacles

    @staticmethod
    def estimate_robot_position(particles: list[Particle]) -> tuple[float, float, float]:
        """
        Calculate the estimated position of the robot based on the passed particles.
        The estimation is based on the mean of the particles.
        :param particles: The particles which represent the possible positions of the robot.
        :return: Returns the estimated position of the robot as a tuple (x, y, yaw)
        """
        x_mean = 0.0
        y_mean = 0.0
        yaw_mean = 0.0
        total_weight = sum(p.weight for p in particles)
        # print(total_weight)

        # Calculate the mean of the particles
        for p in particles:
            x_mean += p.x * p.weight
            y_mean += p.y * p.weight
            yaw_mean += p.yaw * p.weight

        # Normalize the estimated position
        x_mean /= total_weight
        y_mean /= total_weight
        yaw_mean /= total_weight
        yaw_mean = (yaw_mean + np.pi) % (2 * np.pi) - np.pi  # Ensure yaw is between -pi and pi

        return x_mean, y_mean, yaw_mean