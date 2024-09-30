import numpy as np
from sklearn.cluster import KMeans

from src.models.landmark import Landmark
from src.models.particle import Particle
from src.models.point import Point
from src.models.robot import Robot


class InterpretationService:
    """
    This class is responsible for interpreting the fast slam 2.0 algorithm results.
    """

    @staticmethod
    def estimate_robot_position(particles: list[Particle]) -> tuple[float, float, float]:
        """
        Calculate the estimated position of the robot based on the passed particles.
        The estimation is based on the mean of the particles.
        :return: Returns the estimated position of the robot as a tuple (x, y, yaw)
        """
        x_mean = 0.0
        y_mean = 0.0
        yaw_mean = 0.0
        total_weight = sum(p.weight for p in particles)

        # Calculate the mean of the particles
        for p in particles:
            x_mean += p.x * p.weight
            y_mean += p.y * p.weight
            yaw_mean += p.yaw * p.weight

        # Normalize the estimated position
        x_mean /= total_weight
        y_mean /= total_weight
        yaw_mean /= total_weight

        return x_mean, y_mean, yaw_mean

    @staticmethod
    def update_obstacles(existing_obstacles: list[Point], scanned_obstacles: list[Point], robot: Robot):
        """
        Filter out the new obstacles which will be added to the obstacles list so the map will show new borders and obstacles.
        :param existing_obstacles: The existing obstacles list that will be updated with the newly scanned obstacles
        :param scanned_obstacles: The newly scanned obstacles
        :param robot: The robot which scanned the obstacles. The robot's position and orientation will be used to calculate the global coordinates of the scanned obstacles.
        """
        # Apply translation and rotation of the robot to the scanned obstacles to get the global coordinates
        global_points: list[Point] = []
        for scanned_obstacle in scanned_obstacles:
            x_global = robot.x + (
                    scanned_obstacle.x * np.cos(robot.get_yaw_rad()) - scanned_obstacle.y * np.sin(robot.get_yaw_rad()))
            y_global = robot.y + (
                    scanned_obstacle.x * np.sin(robot.get_yaw_rad()) + scanned_obstacle.y * np.cos(robot.get_yaw_rad()))
            global_points.append(Point(x_global, y_global))

        # Round the coordinates of the scanned obstacles to 2 decimal places to add noise to the data.
        # This is important since the laser data is not 100% accurate.
        # Thus, no new obstacles will be added to the same position when scanning the same obstacle multiple times.
        for global_point in global_points:
            global_point.x = round(global_point.x, 2)
            global_point.y = round(global_point.y, 2)

        # Update obstacles list with the scanned obstacles
        existing_coords = {(obstacle.x, obstacle.y) for obstacle in existing_obstacles}
        new_obstacles = [obstacle for obstacle in global_points if
                         (obstacle.x, obstacle.y) not in existing_coords]
        existing_obstacles.extend(new_obstacles)

        return existing_obstacles

    @staticmethod
    def get_weighted_landmarks(particles: list[Particle]) -> list[Landmark]:
        """
        Get the weighted landmark centroids by clustering the landmarks based on the corresponding particle weights using weighted k-means.
        :return: Returns a list with the weighted landmarks
        """
        # Get all landmarks and the corresponding particle weights
        landmark_poses_weights = [(landmark.pose(), particle.weight) for particle in particles for landmark in
                                  particle.landmarks]
        if len(landmark_poses_weights) == 0:
            return []

        (landmark_poses, weights) = zip(*landmark_poses_weights)

        # Get the number of clusters based on average number of landmarks per particle
        num_clusters = round(len(landmark_poses) / len(particles))

        # Use weighted k-means to cluster the landmarks based on the particle weights.
        # (The random state is set for reproducibility of random results which is useful for debugging)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(landmark_poses, sample_weight=weights)

        # Return the cluster centroids which represent the weighted landmarks
        centroids = kmeans.cluster_centers_
        return [Landmark(centroid[0], centroid[1]) for centroid in centroids]
