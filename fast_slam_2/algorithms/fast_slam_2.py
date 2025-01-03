﻿import math
from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy

import numpy as np
from scipy.stats import multivariate_normal

from fast_slam_2.config import NUM_THREAD, NUM_PARTICLES, ROTATION_NOISE, TRANSLATION_NOISE, MEASUREMENT_NOISE
from fast_slam_2.models.landmark import Landmark
from fast_slam_2.models.measurement import Measurement
from fast_slam_2.models.particle import Particle
from fast_slam_2.utils.landmark_utils import LandmarkUtils


class FastSLAM2:
    """
    Class that realizes the FastSLAM 2.0 algorithm.
    """

    def __init__(self):
        """
        Initialize the FastSLAM 2.0 algorithm with the specified number of particles.
        """
        # Initialize particles with the start position of the robot
        self.particles: list[Particle] = [
            Particle(
                x=0.0,
                y=0.0,
                yaw=0.0,
            ) for _ in range(NUM_PARTICLES)
        ]

    def iterate(self, rotation: float, translation: float, measurements: list[Measurement]) -> tuple[
        float, float, float]:
        """
        Perform one iteration of the FastSLAM 2.0 algorithm using the passed translation, rotation, and measurements.
        :param rotation: The performed rotation in radians
        :param translation: The performed translation the robot
        :param measurements: List of measurements to observed landmarks (from the robot's perspective)
        """
        # Move particles in extra threads to speed up the process
        with ThreadPoolExecutor(max_workers=NUM_THREAD) as executor:
            futures = [executor.submit(self.__move_particle, i, rotation, translation) for
                       i in range(NUM_PARTICLES)]
            wait(futures)

        # Update particles (landmarks and weights) in extra threads to speed up the process
        for measurement in measurements:
            with ThreadPoolExecutor(max_workers=NUM_THREAD) as executor:
                futures = [executor.submit(self.__update_particle, particle, measurement) for
                           particle in
                           self.particles]
                wait(futures)

        # Normalize weights
        self.__normalize_weights()

        # Calculate the number of effective particles
        num_effective_particles = self.__calculate_effective_particles()

        # Resample particles if the effective number of particles is less than half of the total number of particles
        if num_effective_particles < NUM_PARTICLES / 2:
            print('\nRESAMPLING')
            self.__low_variance_resample()

        # Return the estimated position of the robot
        return self.__estimate_robot_position()

    def __move_particle(self, index: int, rotation: float, translation: float, ):
        """
        Update the pose of the particle (determined by the passed index) based on the passed translation vector and rotation.
        :param index: The index of the particle in the particle list
        :param translation: The translation vector of the robot
        :param rotation: The rotation angle of the robot in radians
        """
        # Apply uncertainty to the movement of the robot and particles using random Gaussian noise with the standard deviations
        if rotation != 0:
            noisy_translation = 0
            noisy_rotation = rotation + np.random.normal(0, ROTATION_NOISE)
        else:
            noisy_translation = translation + np.random.normal(0, TRANSLATION_NOISE)
            noisy_rotation = 0

        self.particles[index].yaw = (self.particles[index].yaw + noisy_rotation + np.pi) % (
                2 * np.pi) - np.pi  # Ensure yaw stays between -pi and pi
        self.particles[index].x += noisy_translation * np.cos(self.particles[index].yaw)
        self.particles[index].y += noisy_translation * np.sin(self.particles[index].yaw)

    @staticmethod
    def __update_particle(particle: Particle, measurement: Measurement):
        """
        Update the particle based on the passed measurement to an observed landmark.
        A new landmark will be updated if no landmark is associated with the measurement can be found.
        Otherwise, the particle's landmark will be updated based on the passed measurement using and extended Kalman filter.
        The particle weight will be updated based on the likelihood of the measurement.
        :param particle: The particle to update
        :param measurement: The measurement to an observed landmark that should be used to update the particle
        """
        # Search for the associated landmark by comparing the position of the observation and the particle's landmarks
        observed_landmark = Landmark(
            measurement.distance * np.cos(measurement.yaw),
            measurement.distance * np.sin(measurement.yaw)
        )
        associated_landmark, landmark_index = LandmarkUtils.associate_landmarks(observed_landmark, particle.landmarks)

        # If no associated landmark is found, the measurement is referencing to a new landmark
        # and the new landmark will be added to the particle map
        if associated_landmark is None or landmark_index is None:
            landmark_x = particle.x + measurement.distance * math.cos(particle.yaw + measurement.yaw)
            landmark_y = particle.y + measurement.distance * math.sin(particle.yaw + measurement.yaw)
            particle.landmarks.append(Landmark(landmark_x, landmark_y))

        # If an associated landmark is found, the particle's map will be updated based on the actual measurement
        else:
            # Calculate the predicted measurement of the particle and the associated landmark
            dx = associated_landmark.x - particle.x
            dy = associated_landmark.y - particle.y
            q = dx ** 2 + dy ** 2
            distance = math.sqrt(q)
            angle = np.arctan2(dy, dx) - particle.yaw
            predicted_measurement = np.array([distance, angle])

            # Calculate the innovation which is the difference between the actual measurement and the predicted measurement
            innovation = measurement.as_vector() - predicted_measurement
            innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi  # Ensure angle is between -pi and pi

            # Calculate the Jacobian matrix of the particle and the associated landmark.
            # Jacobian describes how changes in the state of the robot influence the measured observations.
            # It helps to link the uncertainties in the estimates with the uncertainties in the measurements
            jacobian = np.array([
                [dx / distance, dy / distance],
                [-dy / q, dx / q]
            ])

            # Calculate the covariance of the observation which depends on the Jacobian matrix,
            # the covariance of the landmark and the measurement noise
            observation_cov = jacobian @ associated_landmark.cov @ jacobian.T + MEASUREMENT_NOISE

            # Calculate the Kalman gain which is used to update the pose/mean and covariance of the associated landmark.
            # It determines how much the actual measurement should be trusted compared to the predicted measurement.
            # Thus, it determines how much the landmark should be updated based on the actual measurement.
            kalman_gain = associated_landmark.cov @ jacobian.T @ np.linalg.inv(observation_cov)

            # Calculate updated pose/mean and covariance of the associated landmark
            mean = associated_landmark.as_vector() + kalman_gain @ innovation
            cov = (np.eye(len(associated_landmark.cov)) - kalman_gain @ jacobian) @ associated_landmark.cov

            # Update the associated landmark
            particle.landmarks[landmark_index] = Landmark(
                x=float(mean[0]),
                y=float(mean[1]),
                cov=cov
            )

            # Calculate the likelihood with the multivariate normal distribution
            likelihood = multivariate_normal.pdf(innovation, mean=np.zeros(len(innovation)), cov=observation_cov)

            # Update the particle weight with the likelihood
            particle.weight *= likelihood

    def __normalize_weights(self):
        """
        Normalizes the weights of all particles.
        :return: Returns the normalized weights as a Nx2 array
        """
        total_weight = sum(p.weight for p in self.particles)

        if total_weight < 1e-5:
            for p in self.particles:
                p.weight = 1.0 / NUM_PARTICLES
        else:
            for p in self.particles:
                p.weight = p.weight if p.weight < 1e-5 else p.weight / total_weight

        return np.array([particle.weight for particle in self.particles])

    def __low_variance_resample(self):
        """
        Resample particles with low variance resampling.
        """
        # Initialize resampling variables
        new_particles = []  # Create a list to store the new particles
        rand_starting_point = np.random.uniform(0, 1 / NUM_PARTICLES)  # Get random starting point
        particle_weight = self.particles[0].weight  # Get weight of the first particle
        particle_index = 0  # Initialize particle index

        # Resample particles
        for m in range(NUM_PARTICLES):
            u = rand_starting_point + m * (1 / NUM_PARTICLES)

            while u > particle_weight:
                particle_index = min(particle_index + 1, NUM_PARTICLES - 1)  # Ensure index is within bounds
                particle_weight += self.particles[particle_index].weight

            # Append the new particles
            new_particles.append(deepcopy(self.particles[particle_index]))

        # Update the particles
        self.particles = new_particles

    def __estimate_robot_position(self) -> tuple[float, float, float]:
        """
        Calculate the estimated position of the robot based on the particles.
        The estimation is based on the mean of the particles.
        :return: Returns the estimated position of the robot as a tuple (x, y, yaw)
        """
        # Get the particle with the biggest weight
        best_particle = max(self.particles, key=lambda p: p.weight)

        return best_particle.x, best_particle.y, best_particle.yaw

    def __calculate_effective_particles(self) -> float:
        """
        Calculate the effective number of particles.
        If the weight of all particles is equal, the effective number of particles is equal to the total number of particles.
        :return: Returns the effective number of particles
        """
        weights = np.array([particle.weight for particle in self.particles])
        total_weight = np.sum(weights ** 2)
        if total_weight < 1 / NUM_PARTICLES:
            return NUM_PARTICLES

        return 1.0 / np.sum(weights ** 2)
