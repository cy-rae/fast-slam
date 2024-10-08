﻿"""
This script is used to plot the map created by the FastSLAM 2.0 algorithm.
The FastSLAM 2.0 algorithm creates a map of the environment using a robot, particles, and landmarks.
The map is stored in a JSON file, which is deserialized in this script.
The robot, particles, and landmarks are then plotted on a 2D grid.
"""
import os
import time

from utils.deserializer import Deserializer
from utils.map_utils import MapUtils

shared_path = "C:/shared"  # Change this path to the path of the shared folder on your machine
file_name = "fast_slam.json"
file_path = os.path.join(shared_path, file_name)

if __name__ == "__main__":
    while True:
        # Deserialize the JSON data that the FastSLAM 2.0 algorithm creates.
        robot, particles, landmarks = Deserializer.deserialize(file_path)

        # If the deserialization is successful, plot the map
        if robot is not None:
            # Plot the robot, landmarks, and particles
            MapUtils.plot_map(robot, particles, landmarks)

        # Wait for a second before updating the map
        time.sleep(1)