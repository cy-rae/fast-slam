import random

import numpy as np

from src.fast_slam_2 import FastSLAM2
from src.models.landmark import Landmark
from src.models.point import Point
from src.models.robot import Robot
from src.services.interpretation_service import InterpretationService
from src.services.map_service import MapService

# Parameters
# The maximum distance between two points to be considered for a neighborhood / cluster
MAXIMUM_POINT_DISTANCE = 2
# The minimum number of points in a neighborhood / cluster
MIN_SAMPLES = 6

# Initialize the FastSLAM 2.0 algorithm with the parameters and environment objects
fast_slam = FastSLAM2(
    # The number of particles used in the FastSLAM 2.0 algorithm
    num_particles=100,
    # Represents the standard deviation of the measurement that depends on the accuracy of the robot's laser sensors. It will be used to calculate the Kalman gain
    measurement_noise=np.array([[0.1, 0.0], [0.0, 0.1]]),
    # Represents the standard deviation of the translation that depends on the accuracy of the robot's odometry sensors
    translation_noise=0.1,
    # Represents the standard deviation of the rotation that depends on the accuracy of the robot's odometry sensors
    rotation_noise=0.1,
    # The maximum distance between two landmarks to be considered for association. Should be the same as the max distance that is used to cluster the landmarks
    max_landmark_distance=MAXIMUM_POINT_DISTANCE
)
robot = Robot() # The robot that will be used to scan the environment and get the measurements to the landmarks. The robot's position will be updated based on the estimated position of the particles.
obstacles: list[Point] = [] # List of obstacles. Only for visualization purposes.
landmarks: list[Landmark] = [] # List of weighted/mean landmarks that will be used to estimate the robot's position.

# The minimum number of iterations before updating the robot's position based on the estimated position of the particles
MIN_ITERATIONS_TO_UPDATE_ROBOT_POSITION = 15
iteration = 0
while True:
    v_i, w_i = random.choice([-1, 0, 1]), random.choice([-1, 0, 1])

    # Get the points of scanned obstacles in the environment using the robot's laser data
    point_list: list[Point] = robot.scan_environment()

    # Update the obstacles list with the scanned points so new borders and obstacles will be added to the map
    obstacles = InterpretationService.update_obstacles(obstacles, point_list, robot)

    # Get the observations of the scanned landmarks
    measurement_list = robot.get_measurements_to_landmarks(point_list, MAXIMUM_POINT_DISTANCE, MIN_SAMPLES)

    # Iterate the FastSLAM 2.0 algorithm with the linear and angular velocities and the measurements to the observed landmarks
    fast_slam.iterate(v_i, w_i, measurement_list)

    # Update the robot's position based on the estimated position of the particles after a configured number of iterations
    if iteration >= MIN_ITERATIONS_TO_UPDATE_ROBOT_POSITION:
        (robot.x, robot.y, robot.yaw) = InterpretationService.estimate_robot_position(fast_slam.particles)
    else:
        # Update the robot's position based on the current linear and angular velocities
        robot.x += v_i * np.cos(robot.get_yaw_rad())
        robot.y += v_i * np.sin(robot.get_yaw_rad())
        robot.yaw = (robot.yaw + w_i) % 360

    # Get the weighted landmarks by clustering the landmarks based on the particle weights
    landmarks = InterpretationService.get_weighted_landmarks(fast_slam.particles)

    # Plot the map with the robot, particles, landmarks and obstacles/borders
    MapService.plot_map(
        robot=robot,
        particles=fast_slam.particles,
        obstacles=obstacles,
        landmarks=landmarks
    )

    # Increase iteration
    iteration += 1
