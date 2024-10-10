import HAL
import numpy as np


class Robot:
    def __init__(self):
        self.prev_x = HAL.getPose3d().x
        self.prev_y = HAL.getPose3d().y
        self.prev_yaw = HAL.getPose3d().yaw

    @staticmethod
    def move() -> tuple[int, int]:
        """
        Set the linear and angular velocity of the robot based on the bumper state.
        :return: Returns Movement.TRANSLATE if the robot moves forward, Movement.ROTATE if the robot rotates
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
                w = 0.5
            else:  # left or center bumper
                w = -0.5

        # If the robot does not hit the wall, the linear and angular velocities will be set to 1 and 0 respectively
        else:
            v = 1
            w = 0

        # Set the linear and angular velocity of the robot
        HAL.setV(v)
        HAL.setW(w)

        return v, w
        # return Movement.TRANSLATE if v == 1 else Movement.ROTATE

    def get_odometry(self, v: float, w: float) -> tuple[float, float]:
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

        if v == 0:
            d_lin = 0
        if w == 0:
            d_ang = 0

        return d_lin, d_ang

robot = Robot()

estimated_x = 0
estimated_y = 0
estimated_yaw = 0
while True:
    v, w = robot.move()

    t, a = robot.get_odometry(v, w)

    estimated_yaw += a
    estimated_x += t * np.cos(estimated_yaw)
    estimated_y += t * np.sin(estimated_yaw)

    print('X: ', HAL.getPose3d().x + 1 - estimated_x)
    print('Y: ', HAL.getPose3d().y - 1.5 - estimated_y)
    print('YAW: ', HAL.getPose3d().yaw - estimated_yaw)