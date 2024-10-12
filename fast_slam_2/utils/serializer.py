import json
import os

from fast_slam_2.models.directed_point import DirectedPoint
from fast_slam_2.models.evaluation_results import EvaluationResults
from fast_slam_2.models.point import Point


class Serializer:
    # Path to the output folder and file name
    shared_path = 'workspace/shared'
    file_name = 'fast_slam.json'
    file_path = os.path.join(shared_path, file_name)

    @staticmethod
    def serialize(
            estimated_robot_pos: DirectedPoint,
            actual_robot_pos: DirectedPoint,
            particles: list[DirectedPoint],
            landmarks: list[Point],
            results: EvaluationResults
    ):
        """
        Serialize the passed robot, particles and landmarks to a JSON serializable dictionary and write the JSON data
        to a file in the shared folder.
        :param estimated_robot_pos: The robot object
        :param actual_robot_pos: The actual robot position
        :param particles: The list of particles
        :param landmarks: The list of landmarks
        :param results: The evaluation results
        """
        json_data = {
            'estimated_robot_pos': estimated_robot_pos.to_dict(),
            'actual_robot_pos': actual_robot_pos.to_dict(),
            'particles': [particle.to_dict() for particle in particles],
            'landmarks': [landmark.to_dict() for landmark in landmarks],
            'results': results.to_dict()
        }

        # Ensure that the shared folder exists
        os.makedirs(Serializer.shared_path, exist_ok=True)

        # Write JSON file
        with open(Serializer.file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
