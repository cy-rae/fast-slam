from numpy import ndarray

from FastSLAM2 import FastSLAM2
from FastSLAM2.models.landmark import Landmark
from FastSLAM2.models.robot import Robot
from FastSLAM2.utils.evaluation_utils import EvaluationUtils
from FastSLAM2.utils.interpreter import Interpreter
from FastSLAM2.utils.landmark_utils import LandmarkUtils
from FastSLAM2.utils.serializer import Serializer

# Initialize the robot, FastSLAM 2.0 algorithm and landmark list
robot = Robot()
fast_slam = FastSLAM2()
landmarks: list[Landmark] = []

# The minimum number of iterations before updating the robot's position based on the estimated position of the particles
while True:
    # Move the robot
    robot.move()

    # Scan the environment using the robot's laser data
    scanned_points: ndarray = robot.scan_environment()

    # Get the translation and rotation of the robot using ICP based on the scanned points and the previous points that the robot has saved.
    translation_vector, rotation = robot.get_transformation(scanned_points)

    # Search for landmarks in the scanned points using line filter and IEPF and get the measurements to them and their points
    measurement_list, landmark_points = LandmarkUtils.get_measurements_to_landmarks(scanned_points)

    # Update the landmark ID in the measurements if they are referencing to an existing landmark
    measurement_list = LandmarkUtils.associate_landmarks(measurement_list, landmark_points)

    # Iterate the FastSLAM2 2.0 algorithm with the linear and angular velocities and the measurements to the observed landmarks
    fast_slam.iterate(translation_vector, rotation, measurement_list)

    # Update the robot's position based on the estimated position of the particles after a configured number of iterations
    (robot.x, robot.y, robot.yaw) = Interpreter.estimate_robot_position(fast_slam.particles)

    # Serialize the robot, particles, and landmarks to a JSON file and store it in the shared folder
    Serializer.plot_map()

    # Validate the robot's position based on the actual position
    EvaluationUtils.evaluate_estimation()