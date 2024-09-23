import GUI
import HAL

import math
import random
import numpy as np
from PIL import Image


def create_particles():
    """
    Create particles with random values.
    """
    array = []
    for i in range(0, num_particles):
        x = random.uniform(-4.1, 5.8)
        y = random.uniform(-4.5, 5.5)
        yaw = random.uniform(0, 359)
        array.append([x, y, yaw])

    return array


def update_particles():
    """
    Update the position of the all particles using the current linear and
    angular velocity of the robot.
    """
    for particle in particles:
        # Update angle
        particle[2] += current_w

        # Update linear position
        yaw_rad = math.radians(particle[2])
        throttled_v = current_v * 0.3  # Throttle velocity because it does not correlate with map
        particle[0] += throttled_v * math.cos(yaw_rad)
        particle[1] += throttled_v * math.sin(yaw_rad)


def update_map(laser_data):
    """
    Update the map by drawing scanned obstacles into the map.
    """
    min_angle = laser_data['minAngle']
    max_angle = laser_data['maxAngle']
    values = laser_data['values']

    # Get the number of laser beams
    num_values = len(values)

    # Angle increment
    angle_increment = (max_angle - min_angle) / num_values

    # Update map for each measurement
    for i, distance in enumerate(values):
        # If the distance is smaller than the minimum range of the laser, no obstacle was found
        if distance < laser_data['minRange']:
            continue

        # Calculate angle of the laser beam
        laser_angle = min_angle + i * angle_increment

        # Conversion to global coordinates
        global_angle = robot_theta + laser_angle  # robot angle + laser angle

        # Calulate obstacle coordinates
        obstacle_x = distance * math.cos(global_angle)
        obstacle_y = distance * math.sin(global_angle)

        # Conversion into map koordinates
        map_x = int(robot_x + obstacle_x / MAP_RESOLUTION)
        map_y = int(robot_y + obstacle_y / MAP_RESOLUTION)

        # Mark obstacle into map
        if 0 <= map_x < MAP_SIZE_X and 0 <= map_y < MAP_SIZE_Y:
            map_grid[map_x, map_y] = 0


def plot_map():
    """
    Store map as an image in the images directory of nginx.
    """
    # Create an image from the random array
    image = Image.fromarray(map_grid)

    # Save the image as a JPG file
    image.save('/usr/share/nginx/html/images/map.jpg')


# Particle variables
num_particles = 10
particles = create_particles()
# GUI.showParticles(particles)

# Weight variables
weights = [0.0] * len(particles)  # Weights of the particles
weight_threshold = 0.2

# Robot variables
current_v = 1
current_w = 1

# Map variables
MAP_SIZE_X = 500
MAP_SIZE_Y = 500
MAP_RESOLUTION = 0.05  # Each cell is 5cm x 5cm
map_grid = np.zeros((MAP_SIZE_X, MAP_SIZE_Y))
# Start position of robot marks origin)
robot_x = MAP_SIZE_X // 2
robot_y = MAP_SIZE_Y // 2
robot_theta = 0

while True:
    laser_data = HAL.getLaserData()
    update_map(laser_data)
    plot_map()

    # Move robot
    # HAL.setV(current_v)
    # HAL.setW(current_w)

    # Update position of particles
    # update_particles()
