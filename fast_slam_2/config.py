"""
This file contains the configuration parameters for the fast_slam_2 2.0 algorithm.
"""
import numpy as np

# Number of particle
NUM_PARTICLES = 100

# Translation and rotation noise represent the standard deviation of the translation and rotation.
# The noise is used to add uncertainty to the movement of the robot and particles.
TRANSLATION_NOISE = 0.0015
ROTATION_NOISE = 0.0015

# The measurement noise of the Kalman filter depends on the laser's accuracy
MEASUREMENT_NOISE = np.array([[0.01, 0.0], [0.0, 0.01]])

# Distance threshold which is used to associate a landmark with an observation
MAXIMUM_LANDMARK_DISTANCE = 2.2