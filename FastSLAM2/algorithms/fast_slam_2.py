import math
import random

import numpy as np
from scipy.stats import multivariate_normal

from FastSLAM2.config import NUM_PARTICLES, TRANSLATION_NOISE, ROTATION_NOISE, MEASUREMENT_NOISE
from FastSLAM2.models.landmark import Landmark
from FastSLAM2.models.measurement import Measurement
from FastSLAM2.models.particle import Particle
from FastSLAM2.models.point import Point


class FastSLAM2:
    """
    Class that realizes the FastSLAM2 2.0 algorithm.
    """

    def __init__(self):
        """
        Initialize the FastSLAM2 2.0 algorithm with the specified number of particles.
        """
        # Initialize particles with the start position of the robot
        self.particles: list[Particle] = [
            Particle(
                x=0.0,
                y=0.0,
                yaw=0.0,
            ) for _ in range(NUM_PARTICLES)
        ]

    def iterate(self, d_linear: float, d_angular: float, measurements: list[Measurement]):
        """
        Perform one iteration of the FastSLAM2 2.0 algorithm using the passed linear and angular delta values and the measurements.
        :param d_linear: linear delta value
        :param d_angular: angular delta value
        :param measurements: list of measurements to observed landmarks (distances and angles of landmark to robot and landmark ID)
        """
        # Update particle poses
        self.__move_particles(d_linear, d_angular)

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
                    # print('\nlikelihood', likelihood)
                    particle.weight *= likelihood
                    # print('nw', particle.weight)

        # Normalisieren der Gewichte
        total_weight = sum(particle.weight for particle in self.particles)
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight

        weights = np.array([particle.weight for particle in self.particles])
        # print('\nNW', weights)

        # Überprüfen der Summe der Gewichte
        if np.sum(weights) == 0:
            raise ValueError("Die Summe der Partikelgewichte ist null! Überprüfen Sie die Gewichtungslogik.")

        # Effektive Partikelzahl
        N_eff = 1.0 / np.sum(weights ** 2)

        # Resampling nur durchführen, wenn N_eff klein ist
        if N_eff < len(self.particles) / 2:
            self.particles = self.__systematic_resample()

    def __move_particles(self, d_linear: float, d_angular: float):
        """
        Update the poses of the particles based on the passed linear and angular delta values.
        :param d_linear: linear delta value
        :param d_angular: angular delta value
        """
        # Apply uncertainty to the movement of the robot and particles using random Gaussian noise with the standard deviations
        d_linear += random.gauss(0, TRANSLATION_NOISE)
        d_angular += random.gauss(0, ROTATION_NOISE)

        for p in self.particles:
            p.yaw = (p.yaw + d_angular + np.pi) % (2 * np.pi) - np.pi  # Ensure yaw stays between -pi and pi
            p.x += d_linear * np.cos(p.yaw)
            p.y += d_linear * np.sin(p.yaw)

    def __systematic_resample(self):
        """
        Perform systematic resampling of the particles.
        :return: Returns the resampled particles
        """
        num_particles = len(self.particles)

        # Normalisierung der Gewichte
        weights = np.array([particle.weight for particle in self.particles])
        weights /= np.sum(weights)  # Normalisierung der Gewichte

        # Resampling
        positions = (np.arange(num_particles) + np.random.rand()) / num_particles
        cumulative_sum = np.cumsum(weights)

        # Resampling mit Indices
        indices = np.searchsorted(cumulative_sum, positions)

        # Erstellen der neuen Partikel basierend auf den Indizes
        resampled_particles = [self.particles[i] for i in indices]

        return resampled_particles