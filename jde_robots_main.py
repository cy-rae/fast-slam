from numpy import ndarray

from FastSLAM2 import FastSLAM2
from FastSLAM2 import Measurement
from FastSLAM2 import Robot
from FastSLAM2 import EvaluationUtils
from FastSLAM2 import LandmarkUtils
from FastSLAM2 import Serializer

# Initialize the robot, FastSLAM 2.0 algorithm and landmark list
robot = Robot()
fast_slam = FastSLAM2()

# The minimum number of iterations before updating the robot's position based on the estimated position of the particles
while True:
    # Move the robot
    robot.move()

    # Scan the environment using the robot's laser data
    scanned_points: ndarray = robot.scan_environment()

    # Get the translation and rotation of the robot using ICP based on the scanned points and the previous points that the robot has saved.
    translation_vector, rotation = robot.get_transformation(scanned_points)

    # Search for landmarks in the scanned points using line filter and hough transformation and get the measurements to them
    measurement_list: list[Measurement] = LandmarkUtils.get_measurements_to_landmarks(scanned_points)

    # Iterate the FastSLAM2 2.0 algorithm with the linear and angular velocities and the measurements to the observed landmarks
    # and estimate the position of the robot based on the particles.
    (robot.x, robot.y, robot.yaw) = fast_slam.iterate(translation_vector, rotation, measurement_list)

    # Serialize the robot, particles, and landmarks to a JSON file and store it in the shared folder
    Serializer.serialize(robot, fast_slam.particles, LandmarkUtils.known_landmarks)

    # Validate the robot's position based on the actual position
    EvaluationUtils.evaluate_estimation(robot)
