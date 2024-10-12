import math
from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy

import numpy as np
from scipy.stats import multivariate_normal

# from fast_slam_2.config import NUM_PARTICLES, TRANSLATION_NOISE, ROTATION_NOISE, MEASUREMENT_NOISE, NUM_CORES
from fast_slam_2.models.landmark import Landmark
from fast_slam_2.models.measurement import Measurement
from fast_slam_2.models.particle import Particle
from fast_slam_2.utils.landmark_utils import LandmarkUtils


class FastSLAM2:
    """
    Class that realizes the fast_slam_2 2.0 algorithm.
    """

    def __init__(self, num_particles: int):
        """
        Initialize the fast_slam_2 2.0 algorithm with the specified number of particles.
        """
        # Initialize particles with the start position of the robot
        self.particles: list[Particle] = [
            Particle(
                x=0.0,
                y=0.0,
                yaw=0.0,
            ) for _ in range(num_particles)
        ]

    def iterate(self, d_lin: float, rotation: float, measurements: list[Measurement],
                TRANSLATION_NOISE: float, ROTATION_NOISE: float, MEASUREMENT_NOISE: np.ndarray, NUM_PARTICLES: int,
                NUM_CORES: int
                ) -> tuple[float, float, float]:
        """
        Perform one iteration of the fast_slam_2 2.0 algorithm using the passed translation, rotation, and measurements.
        :param d_lin: The translation vector of the robot
        :param rotation: The rotation angle of the robot in radians
        :param measurements: List of measurements to observed landmarks (distances and angles of landmark to robot and landmark ID)
        """
        # Move particles in extra threads to speed up the process
        with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
            futures = [executor.submit(self.__move_particle, i, d_lin, rotation, TRANSLATION_NOISE, ROTATION_NOISE) for
                       i in range(NUM_PARTICLES)]
            wait(futures)

        # Update particles (landmarks and weights) in extra threads to speed up the process
        for measurement in measurements:
            with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
                futures = [executor.submit(self.__update_particle, particle, measurement, MEASUREMENT_NOISE) for
                           particle in
                           self.particles]
                wait(futures)

        # Normalize weights and resample particles
        self.__normalize_weights(NUM_PARTICLES)

        # Calculate the number of effective particles
        num_effective_particles = self.__calculate_effective_particles(NUM_PARTICLES)

        # Resample particles if the effective number of particles is less than half of the total number of particles
        if num_effective_particles < NUM_PARTICLES / 2:
            print('RESAMPLING')
            self.__low_variance_resample(NUM_PARTICLES)

        # Return the estimated position of the robot
        return self.__estimate_robot_position()

    def __move_particle(self, index: int, d_lin: float, rotation: float, TRANSLATION_NOISE: float,
                        ROTATION_NOISE: float):
        """
        Update the particle (determined by the passed index) based on the passed translation vector and rotation.
        :param index: The index of the particle in the particle list
        :param d_lin: The translation vector of the robot
        :param rotation: The rotation angle of the robot in radians
        """
        # Apply uncertainty to the movement of the robot and particles using random Gaussian noise with the standard deviations
        d_lin += np.random.normal(0, TRANSLATION_NOISE)
        rotation += np.random.normal(0, ROTATION_NOISE)

        self.particles[index].yaw = (self.particles[index].yaw + rotation + np.pi) % (
                    2 * np.pi) - np.pi  # Ensure yaw stays between -pi and pi
        self.particles[index].x += d_lin * np.cos(self.particles[index].yaw)
        self.particles[index].y += d_lin * np.sin(self.particles[index].yaw)

    @staticmethod
    def __update_particle(particle: Particle, measurement: Measurement, MEASUREMENT_NOISE: np.ndarray):
        # Search for the associated landmark by comparing the position of the observation and the particle's landmarks
        observed_landmark = Landmark(
            measurement.landmark_id,
            measurement.distance * np.cos(measurement.yaw),
            measurement.distance * np.sin(measurement.yaw)
        )
        associated_landmark, landmark_index = LandmarkUtils.associate_landmarks(observed_landmark, particle.landmarks)

        # If no associated landmark is found, the measurement is referencing to a new landmark
        # and the new landmark will be added to the particle map
        if associated_landmark is None or landmark_index is None:
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
            particle.landmarks[landmark_index] = Landmark(
                identifier=associated_landmark.id,
                x=float(mean[0]),
                y=float(mean[1]),
                cov=cov
            )

            # Calculate the likelihood with the multivariate normal distribution
            likelihood = multivariate_normal.pdf(innovation, mean=np.zeros(len(innovation)), cov=observation_cov)

            # Update the particle weight with the likelihood
            particle.weight *= likelihood

    def __normalize_weights(self, NUM_PARTICLES: int):
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

    def __low_variance_resample(self, NUM_PARTICLES: int):
        """
        Resample particles with low variance resampling.
        :param weights: The normalized weights of the particles
        :return:
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
        x_mean = 0.0
        y_mean = 0.0
        yaw_mean = 0.0
        total_weight = sum(p.weight for p in self.particles)

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

    def __calculate_effective_particles(self, NUM_PARTICLES: int) -> float:
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
