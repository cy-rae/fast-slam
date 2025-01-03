﻿"""
This file contains the configuration parameters for the fast_slam_2 2.0 algorithm.
"""
import numpy as np

# Number of particle
NUM_PARTICLES = 20

# Translation and rotation noise represent the standard deviation of the translation and rotation.
# The noise is used to add uncertainty to the movement of the robot and particles.
TRANSLATION_NOISE = 0.0055
ROTATION_NOISE = 0.001

# The measurement noise of the Kalman filter depends on the laser's accuracy
MEASUREMENT_NOISE = np.array([[0.001, 0.0], [0.0, 0.001]])

# Distance threshold which is used to associate a landmark with an observation. The uncertainty is in mahalanobis distance.
MAXIMUM_LANDMARK_DISTANCE = 8

# Number of cores used for parallel updating of particles
NUM_THREAD = 20
