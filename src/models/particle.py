import numpy as np

from src.models.directed_point import DirectedPoint
from src.models.landmark import Landmark
from src.models.measurement import Measurement
from src.models.point import Point


class Particle(DirectedPoint):
    """
    Class to represent a particle in the FastSLAM 2.0 algorithm.
    """

    def __init__(self, x: float, y: float, yaw: float, weight: float):
        super().__init__(x, y, yaw)
        self.weight = weight
        self.landmarks: list[Landmark] = []

    def get_associated_landmark(self, measurement: Measurement, max_distance: float) -> int or None:
        """
        Search for a landmark in the landmarks list that is associated with the observation that is described by the
        passed measurement using mahalanobis distance.
        :param measurement: The measurement of an observed landmark (distance and angle) to the robot.
        :param max_distance: The maximum distance between two landmarks to be considered for association.
        :return: Returns None if no landmark can be found. Else, the index of the associated landmark will be returned.
        """
        # Get the pose of the observed landmark starting from the particle
        observed_landmark_pose = np.array([
            self.x + measurement.distance * np.cos(self.get_yaw_rad() + measurement.yaw),
            self.y + measurement.distance * np.sin(self.get_yaw_rad() + measurement.yaw)
        ])

        for particle_landmark in self.landmarks:
            # Calculate the mahalanobis distance between the observed landmark and the particle's landmark
            # using the covariance matrix of the particle's landmark
            distance = Point.mahalanobis_distance(
                particle_landmark.pose(),
                observed_landmark_pose,
                particle_landmark.cov
            )

            # Use cluster radius as threshold for association
            if distance < max_distance:
                return self.landmarks.index(particle_landmark)

        return None
