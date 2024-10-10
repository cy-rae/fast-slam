import HAL
import numpy as np


class Robot:
    def __init__(self):
        self.prev_x = HAL.getPose3d().x
        self.prev_y = HAL.getPose3d().y
        self.prev_yaw = HAL.getPose3d().yaw


    def get_odometry(self) -> tuple[float, float]:
        """
        Get the linear and angular displacement of the robot based on the linear and angular velocity.
        :return: Returns the linear and angular displacement of the robot as a tuple (d_lin, d_ang)
        """
        # Get the current pose of the robot
        curr_x = HAL.getPose3d().x
        curr_y = HAL.getPose3d().y
        curr_yaw = HAL.getPose3d().yaw

        # Calculate the linear and angular displacement of the robot
        d_lin = np.sqrt((curr_x - self.prev_x) ** 2 + (curr_y - self.prev_y) ** 2)
        d_ang = curr_yaw - self.prev_yaw

        # Add noise to the linear and angular displacement to simulate the odometry noise
        # d_lin += np.random.normal(0, 0.0015)
        # d_ang += np.random.normal(0, 0.0015)

        # Update the previous pose of the robot
        self.prev_x = curr_x
        self.prev_y = curr_y
        self.prev_yaw = curr_yaw

        return d_lin, d_ang

robot = Robot()

estimated_x = 0
estimated_y = 0
estimated_yaw = 0
while True:
    HAL.setV(1)

    v, a = robot.get_odometry()

    estimated_yaw += a
    estimated_x += v * np.cos(estimated_yaw)
    estimated_y += v * np.sin(estimated_yaw)

    print('X: ', HAL.getPose3d().x + 1 - estimated_x)
    print('Y: ', HAL.getPose3d().y - 1.5 - estimated_y)
    print('YAW: ', HAL.getPose3d().yaw - estimated_yaw)