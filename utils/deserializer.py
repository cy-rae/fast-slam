import os

from FastSLAM2.models.directed_point import DirectedPoint
from FastSLAM2.models.landmark import Landmark
from FastSLAM2.models.point import Point
from FastSLAM2.models.robot import Robot


class Deserializer:
    """
    Class to deserialize the JSON data that produces the FastSLAM 2.0 algorithm into classes.
    """

    @staticmethod
    def deserialize(file_path: str) -> tuple[DirectedPoint or None, list[DirectedPoint], list[Point]]:
        """
        Deserialize the JSON data into classes.
        :return: Returns the robot, landmarks, and particles
        """
        # Check if file exists
        if not file_path or not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            return None, [], []

        # Read the JSON data from the file
        with open(file_path, 'r') as file:
            json_data: str = file.read()

            robot = Robot.from_dict(json_data["robot"])
            particles = Deserializer.__get_particles(json_data)
            landmarks = Deserializer.__get_landmarks(json_data)

        return robot, particles, landmarks

    @staticmethod
    def __get_particles(json_data: str) -> list[DirectedPoint]:
        """
        Deserialize the JSON data into a list of directed point
        :param json_data: The JSON data
        :return: Returns the list of directed points which represent the particles
        """
        particles = []
        for data in json_data["particles"]:
            particle = DirectedPoint.from_dict(data)
            particles.append(particle)

        return particles

    @staticmethod
    def __get_landmarks(json_data: str) -> list[Landmark]:
        """
        Deserialize the JSON data into a list of points
        :param json_data: The JSON data
        :return: Returns the list of points which represent the landmarks
        """
        landmarks = []
        for data in json_data["landmarks"]:
            landmark = Point.from_dict(data)
            landmarks.append(landmark)

        return landmarks
