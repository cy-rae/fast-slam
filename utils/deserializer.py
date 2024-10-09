import json
import os

from exceptiongroup import catch


class Deserializer:
    """
    Class to deserialize the JSON data that produces the FastSLAM 2.0 algorithm into classes.
    """

    @staticmethod
    def deserialize(file_path: str) -> tuple[
        tuple[float, float, float] or None,
        list[tuple[float, float, float]],
        list[tuple[float, float]]
    ]:
        """
        Deserialize the JSON data into classes.
        :return: Returns the robot, landmarks, and particles as tuples and lists
        """
        # Check if file exists
        if not file_path or not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            return None, [], []

        # Read the JSON data from the file
        with open(file_path, 'r') as file:
            try:
                json_data: dict = json.load(file)
            except Exception as e:
                return None, [], []

            robot: tuple[float, float, float] = Deserializer.__deserialize_directed_point(json_data['robot'])
            particles: list[tuple[float, float, float]] = Deserializer.__deserialize_directed_points(
                json_data['particles'])
            landmarks: list[tuple[float, float]] = Deserializer.__deserialize_points(json_data['landmarks'])

        return robot, particles, landmarks

    @staticmethod
    def __deserialize_directed_point(data: dict) -> tuple[float, float, float]:
        """
        Deserialize a directed point from the passed data
        :param data: This data contains the properties of the directed point
        :return: Returns the x, y, and yaw values of the directed point
        """
        x = float(data['x'])
        y = float(data['y'])
        yaw = float(data['yaw'])

        return x, y, yaw

    @staticmethod
    def __deserialize_directed_points(json_list: dict) -> list[tuple[float, float, float]]:
        """
        Deserialize the JSON data into a list of directed point
        :param json_list: The JSON list
        :return: Returns the list of directed points represented as a tuple (x, y, yaw)
        """
        directed_points: list[tuple[float, float, float]] = []
        for data in json_list:
            directed_point: tuple[float, float, float] = Deserializer.__deserialize_directed_point(data)
            directed_points.append(directed_point)

        return directed_points

    @staticmethod
    def __deserialize_point(data: dict) -> tuple[float, float]:
        """
        Deserialize a point from the passed data
        :param data: This data contains the properties of the point
        :return: Returns the x and y values of the point
        """
        x = float(data['x'])
        y = float(data['y'])

        return x, y

    @staticmethod
    def __deserialize_points(json_list: dict) -> list[tuple[float, float]]:
        """
        Deserialize the JSON data into a list of points
        :param json_list: The JSON list
        :return: Returns the list of directed points represented as a tuple (x, y, yaw)
        """
        points: list[tuple[float, float]] = []
        for data in json_list:
            point: tuple[float, float] = Deserializer.__deserialize_point(data)
            points.append(point)

        return points
