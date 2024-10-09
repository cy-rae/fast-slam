import json
import os

from fast_slam_2.models.landmark import Landmark
from fast_slam_2.models.particle import Particle
from fast_slam_2.models.robot import Robot


class Serializer:
    # Path to the output folder and file name
    shared_path = 'workspace/shared'
    file_name = 'fast_slam.json'
    file_path = os.path.join(shared_path, file_name)

    @staticmethod
    def serialize(robot: Robot, particles: list[Particle], landmarks: list[Landmark]):
        """
        Serialize the passed robot, particles and landmarks to a JSON serializable dictionary and write the JSON data
        to a file in the shared folder.
        :param robot: The robot object
        :param particles: The list of particles
        :param landmarks: The list of landmarks
        """
        json_data = {
            'robot': robot.to_dict(),
            'particles': [particle.to_dict() for particle in particles],
            'landmarks': [landmark.to_dict() for landmark in landmarks]
        }

        # Ensure that the shared folder exists
        os.makedirs(Serializer.shared_path, exist_ok=True)

        # Write JSON file
        with open(Serializer.file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
