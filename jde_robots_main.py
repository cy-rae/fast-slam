import time

import numpy as np
from numpy import ndarray

from fast_slam_2 import EvaluationUtils
from fast_slam_2 import FastSLAM2
from fast_slam_2 import LandmarkUtils
from fast_slam_2 import Measurement
from fast_slam_2 import Robot
from fast_slam_2 import Serializer
from fast_slam_2 import ObstacleUtils

# Number of particle
NUM_PARTICLES = 20

# Translation and rotation noise represent the standard deviation of the translation and rotation.
# The noise is used to add uncertainty to the movement of the robot and particles.
TRANSLATION_NOISE = 0.0055
ROTATION_NOISE = 0.001

# The measurement noise of the Kalman filter depends on the laser's accuracy
MEASUREMENT_NOISE = np.array([[0.001, 0.0], [0.0, 0.001]])

# Number of cores used for parallel updating of particles
NUM_CORES = 28





# Initialize the robot, FastSLAM 2.0 algorithm and landmark list
robot = Robot()
fast_slam = FastSLAM2(NUM_PARTICLES)

MIN_ITERATIONS = 150
iteration = 0
while True:
    # First initialize the evaluation utils. The process will not start until the robot has fully initialized.
    if not EvaluationUtils.initialized:
        EvaluationUtils.try_to_initialize()
        continue

    # Move the robot
    v, w = robot.move(0.3, 0.5)

    # Scan the environment using the robot's laser data
    scanned_points: ndarray = robot.scan_environment()

    # Update the known obstacles with the scanned points
    ObstacleUtils.update_obstacles(scanned_points, robot)

    # Get the translation and rotation of the robot using ICP based on the scanned points and the previous points that the robot has saved.
    d_ang, d_lin = robot.get_displacement(v, w)
    # d_ang, d_lin= robot.get_transformation(scanned_points, v, w)
    # d_ang, d_lin = robot.get_rotation_and_translation(scanned_points, v, w)

    # Search for landmarks in the scanned points using line filter and hough transformation and get the measurements to them
    measurement_list: list[Measurement] = LandmarkUtils.get_measurements_to_landmarks(scanned_points)

    # Iterate the fast_slam_2 2.0 algorithm with the linear and angular velocities and the measurements to the observed landmarks
    # and estimate the position of the robot based on the particles.
    # If the iteration is less than the minimum iterations, the robot will be updated by the displacement of the robot.

    (x, y, yaw) = fast_slam.iterate(
        d_lin,
        d_ang,
        measurement_list,
        TRANSLATION_NOISE,
        ROTATION_NOISE,
        MEASUREMENT_NOISE,
        NUM_PARTICLES,
        NUM_CORES
    )

    if iteration < MIN_ITERATIONS:
        robot.yaw = (robot.yaw + d_ang + np.pi) % (2 * np.pi) - np.pi
        robot.x = robot.x + d_lin * np.cos(robot.yaw)
        robot.y = robot.y + d_lin * np.sin(robot.yaw)
    else:
        (robot.x, robot.y, robot.yaw) = (x, y, yaw)

    # Update the known landmarks with the observed landmarks
    LandmarkUtils.update_known_landmarks(fast_slam.particles)

    # Evaluate the robot's position based on the actual position
    results, actual_pos = EvaluationUtils.evaluate_estimation(robot)

    # Serialize the robot, particles, and landmarks to a JSON file and store it in the shared folder
    Serializer.serialize(robot, actual_pos, fast_slam.particles, LandmarkUtils.known_landmarks, results)
    Serializer.serialize_obstacles(ObstacleUtils.obstacles)

    # time.sleep(0.2)
    iteration += 1