import numpy as np
from numpy import ndarray

from fast_slam_2 import EvaluationUtils
from fast_slam_2 import FastSLAM2
from fast_slam_2 import LandmarkUtils
from fast_slam_2 import Measurement
from fast_slam_2 import Robot
from fast_slam_2 import Serializer

# Initialize the robot, FastSLAM 2.0 algorithm and landmark list
robot = Robot()
fast_slam = FastSLAM2()

# The minimum number of iterations before updating the robot's position based on the estimated position of the particles
MIN_ITERATIONS = 200
i = 0
while True:
    # Move the robot
    v, w = robot.move(0.5, 0.5)

    # Scan the environment using the robot's laser data
    scanned_points: ndarray = robot.scan_environment()

    # Get the translation and rotation of the robot using ICP based on the scanned points and the previous points that the robot has saved.
    # d_ang, d_lin = robot.get_displacement(v, w)
    d_ang, d_lin= robot.get_transformation(scanned_points, v, w)

    # Search for landmarks in the scanned points using line filter and hough transformation and get the measurements to them
    measurement_list: list[Measurement] = LandmarkUtils.get_measurements_to_landmarks(scanned_points)

    # Iterate the fast_slam_2 2.0 algorithm with the linear and angular velocities and the measurements to the observed landmarks
    # and estimate the position of the robot based on the particles.
    (x, y, yaw) = fast_slam.iterate(d_lin, d_ang, measurement_list)

    if i < MIN_ITERATIONS:
        robot.yaw = (robot.yaw + d_ang + np.pi) % (2 * np.pi) - np.pi
        robot.x += d_lin * np.cos(robot.yaw)
        robot.y += d_lin * np.sin(robot.yaw)
    else:
        robot.x = x
        robot.y = y
        robot.yaw = yaw

    # Serialize the robot, particles, and landmarks to a JSON file and store it in the shared folder
    Serializer.serialize(robot, fast_slam.particles, LandmarkUtils.known_landmarks)

    # Validate the robot's position based on the actual position
    EvaluationUtils.evaluate_estimation(robot)

    i += 1
