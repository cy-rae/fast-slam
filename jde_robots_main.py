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

    # Get the translation and rotation of the robot based on the control commands (linear and angular velocities).
    rotation, translation = robot.get_transformation(v, w)

    # Search for landmarks in the scanned points using line filter and hough transformation and get the measurements to them
    measurement_list: list[Measurement] = LandmarkUtils.get_measurements_to_landmarks(scanned_points)

    # Iterate the fast_slam_2 2.0 algorithm with the linear and angular velocities and the measurements to the observed landmarks
    # and estimate the position of the robot based on the particles.
    (x, y, yaw) = fast_slam.iterate(rotation, translation, measurement_list)

    # If the iteration is less than the minimum iterations, the robot will be updated by the displacement of the robot.
    if iteration < MIN_ITERATIONS:
        robot.yaw = (robot.yaw + rotation + np.pi) % (2 * np.pi) - np.pi
        robot.x = robot.x + translation * np.cos(robot.yaw)
        robot.y = robot.y + translation * np.sin(robot.yaw)
    else:
        (robot.x, robot.y, robot.yaw) = (x, y, yaw)

    # Update the known landmarks with the observed landmarks
    LandmarkUtils.update_known_landmarks(fast_slam.particles)

    # Evaluate the robot's position based on the actual position
    results, actual_pos = EvaluationUtils.evaluate_estimation(robot)

    # Serialize the robot, particles, and landmarks to a JSON file and store it in the shared folder
    Serializer.serialize(robot, actual_pos, fast_slam.particles, LandmarkUtils.known_landmarks, results)

    # time.sleep(0.2)
    iteration += 1