import numpy as np
from numpy import ndarray

from fast_slam_2 import EvaluationUtils
from fast_slam_2 import FastSLAM2
from fast_slam_2 import LandmarkUtils
from fast_slam_2 import Measurement
from fast_slam_2 import Robot
from fast_slam_2 import Serializer

# Number of particle
NUM_PARTICLES = 30

# Translation and rotation noise represent the standard deviation of the translation and rotation.
# The noise is used to add uncertainty to the movement of the robot and particles.
TRANSLATION_NOISE = 0.005
ROTATION_NOISE = 0.004

# The measurement noise of the Kalman filter depends on the laser's accuracy
MEASUREMENT_NOISE = np.array([[0.0001, 0.0], [0.0, 0.0001]])

# Number of cores used for parallel updating of particles
NUM_CORES = 28





# Initialize the robot, FastSLAM 2.0 algorithm and landmark list
robot = Robot()
fast_slam = FastSLAM2(NUM_PARTICLES)

while True:
    # First initialize the evaluation utils. The process will not start until the robot has fully initialized.
    if not EvaluationUtils.initialized:
        EvaluationUtils.try_to_initialize()
        continue

    # Move the robot
    v, w = robot.move(0.3, 0.4)

    # Scan the environment using the robot's laser data
    scanned_points: ndarray = robot.scan_environment()

    # Get the translation and rotation of the robot using ICP based on the scanned points and the previous points that the robot has saved.
    d_ang, d_lin = robot.get_displacement(v, w)
    # d_ang, d_lin= robot.get_transformation(scanned_points, v, w)
    # d_ang, d_lin = robot.get_rotation_and_translation(scanned_points, v, w)

    # Search for landmarks in the scanned points using line filter and hough transformation and get the measurements to them
    measurement_list: list[Measurement] = LandmarkUtils.get_measurements_to_landmarks(scanned_points)

    # Iterate the fast_slam_2 2.0 algorithm with the linear and angular velocities and the measurements to the observed landmarks
    # and estimate the position of the robot based on the particles.
    (robot.x, robot.y, robot.yaw) = fast_slam.iterate(
        d_lin,
        d_ang,
        measurement_list,
        TRANSLATION_NOISE,
        ROTATION_NOISE,
        MEASUREMENT_NOISE,
        NUM_PARTICLES,
        NUM_CORES
    )

    # Update the known landmarks with the observed landmarks
    LandmarkUtils.update_known_landmarks(fast_slam.particles)

    # Evaluate the robot's position based on the actual position
    results, actual_pos = EvaluationUtils.evaluate_estimation(robot)

    # Serialize the robot, particles, and landmarks to a JSON file and store it in the shared folder
    Serializer.serialize(robot, actual_pos, fast_slam.particles, LandmarkUtils.known_landmarks, results)