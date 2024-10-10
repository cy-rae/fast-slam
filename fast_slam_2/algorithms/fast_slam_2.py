import math
import random

import numpy as np
from numpy import ndarray
from scipy.stats import multivariate_normal

from fast_slam_2.config import NUM_PARTICLES, TRANSLATION_NOISE, ROTATION_NOISE, MEASUREMENT_NOISE
from fast_slam_2.models.landmark import Landmark
from fast_slam_2.models.measurement import Measurement
from fast_slam_2.models.particle import Particle


class FastSLAM2:
    """
    Class that realizes the fast_slam_2 2.0 algorithm.
    """

    def __init__(self):
        """
        Initialize the fast_slam_2 2.0 algorithm with the specified number of particles.
        """
        # Initialize particles with the start position of the robot
        self.particles: list[Particle] = [
            Particle(
                x=0.0,
                y=0.0,
                yaw=0.0,
            ) for _ in range(NUM_PARTICLES)
        ]

    def iterate(self, d_lin: float, rotation: float, measurements: list[Measurement]):
        """
        Perform one iteration of the fast_slam_2 2.0 algorithm using the passed translation, rotation, and measurements.
        :param d_lin: The translation vector of the robot
        :param rotation: The rotation angle of the robot in radians
        :param measurements: List of measurements to observed landmarks (distances and angles of landmark to robot and landmark ID)
        """
        # Update particle poses
        self.__move_particles(d_lin, rotation)

        # Update particles (landmarks and weights)
        for measurement in measurements:
            for particle in self.particles:

                # Search for the associated landmark by the landmark ID of the measurement
                associated_landmark, associated_landmark_index = particle.get_landmark(measurement.landmark_id)

                # If no associated landmark is found, the measurement is referencing to a new landmark
                # and the new landmark will be added to the particle map
                if associated_landmark is None or associated_landmark_index is None:
                    landmark_x = particle.x + measurement.distance * math.cos(particle.yaw + measurement.yaw)
                    landmark_y = particle.y + measurement.distance * math.sin(particle.yaw + measurement.yaw)
                    particle.landmarks.append(Landmark(measurement.landmark_id, landmark_x, landmark_y))

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
                    particle.landmarks[associated_landmark_index] = Landmark(
                        identifier=associated_landmark.id,
                        x=float(mean[0]),
                        y=float(mean[1]),
                        cov=cov
                    )

                    # Calculate the likelihood with the multivariate normal distribution
                    likelihood = multivariate_normal.pdf(innovation, mean=np.zeros(len(innovation)), cov=observation_cov)

                    # Update the particle weight with the likelihood
                    particle.weight *= likelihood

        # Normalize weights and resample particles
        weights = self.__get_normalized_weights()

        # Calculate the number of effective particles
        N_eff = 1.0 / np.sum(weights ** 2)

        # Resample particles if the effective number of particles is less than half of the total number of particles
        if N_eff < NUM_PARTICLES / 2:
            print('RESAMPLING')
            self.__low_variance_resample(weights)

        # Return the estimated position of the robot
        return self.__estimate_robot_position()

    def __move_particles(self, d_lin: float, rotation: float):
        """
        Update the poses of the particles based on the passed translation vector and rotation.
        :param d_lin: The translation vector of the robot
        :param rotation: The rotation angle of the robot in radians
        """
        for p in self.particles:
            # Apply uncertainty to the movement of the robot and particles using random Gaussian noise with the standard deviations
            d_lin += np.random.normal(0, TRANSLATION_NOISE)
            rotation += np.random.normal(0, ROTATION_NOISE)

            p.yaw = (p.yaw + rotation + np.pi) % (2 * np.pi) - np.pi  # Ensure yaw stays between -pi and pi
            p.x += d_lin * np.cos(p.yaw)
            p.y += d_lin * np.sin(p.yaw)


    def __get_normalized_weights(self):
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
                p.weight = 0 if p.weight < 1e-5 else p.weight / total_weight

        return np.array([particle.weight for particle in self.particles])

    def __low_variance_resample(self, weights: ndarray):
        """
        Resample particles with low variance resampling.
        :param weights: The normalized weights of the particles
        :return:
        """
        # Create a list to store the new particles
        new_particles = []

        # Get random starting point
        rand_starting_point = np.random.uniform(0, 1 / NUM_PARTICLES)

        # Get index of the first particle
        particle_weight = weights[0]
        particle_index = 0

        # Calculate the step size
        step = 1 / NUM_PARTICLES

        # Resample particles
        for m in range(NUM_PARTICLES):
            u = rand_starting_point + m * step
            while u > particle_weight and particle_index < NUM_PARTICLES:
                particle_weight += weights[particle_index]
                particle_index += 1
            new_particles.append(self.particles[particle_index])

        # Update the particles
        self.particles = new_particles
        print('Resampled particles:', len(self.particles))

    def __systematic_resample(self, weights: ndarray) -> list[Particle]:
        """
        Perform systematic resampling of the particles.
        :param weights: The normalized weights of the particles
        :return: Returns the resampled particles
        """
        # Resampling
        positions = (np.arange(NUM_PARTICLES) + np.random.rand()) / NUM_PARTICLES
        cumulative_sum = np.cumsum(weights)

        # Resampling mit Indices
        indices = np.searchsorted(cumulative_sum, positions)

        # Erstellen der neuen Partikel basierend auf den Indizes
        resampled_particles = [self.particles[i] for i in indices]
        print('Resampled particles:', len(resampled_particles))

        return resampled_particles

    def __estimate_robot_position(self) -> tuple[float, float, float]:
        """
        Calculate the estimated position of the robot based on the particles.
        The estimation is based on the mean of the particles.
        :return: Returns the estimated position of the robot as a tuple (x, y, yaw)
        """
        x_mean = 0.0
        y_mean = 0.0
        yaw_mean = 0.0
        total_weight = sum(p.weight for p in self.particles)
        # print(total_weight)

        # Calculate the mean of the particles
        for p in self.particles:
            x_mean += p.x * p.weight
            y_mean += p.y * p.weight
            yaw_mean += p.yaw * p.weight

        # Normalize the estimated position
        x_mean /= total_weight
        y_mean /= total_weight
        yaw_mean /= total_weight
        yaw_mean = (yaw_mean + np.pi) % (2 * np.pi) - np.pi  # Ensure yaw is between -pi and pi

        return x_mean, y_mean, yaw_mean