import HAL
import numpy as np


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
        v = 0.5
        w = 0

    # Set the linear and angular velocity of the robot
    HAL.setV(v)
    HAL.setW(w)

    return v, w


prev_timestamp = HAL.getLaserData().timeStamp
x = 0
y = 0
yaw = 0

while True:
    v, w = move()

    timestamp = HAL.getLaserData().timeStamp
    delta_t = timestamp - prev_timestamp
    prev_timestamp = timestamp

    # Calculate the displacement of the robot
    d_ang = w * delta_t
    d_lin = v * delta_t * 0.6

    # Update the position of the robot
    yaw = (yaw + d_ang + np.pi) % (2 * np.pi) - np.pi  # Ensure yaw stays between -pi and pi
    x += d_lin * np.cos(yaw)
    y += d_lin * np.sin(yaw)

    # Print the updated position of the robot
    print('\nYAW', yaw)
    print('ACTUAL YAW', HAL.getPose3d().yaw)

    print('\nX', x)
    print('ACTUAL X', HAL.getPose3d().x + 1)

    print('\nY', y)
    print('ACTUAL Y', HAL.getPose3d().y - 1.5)
