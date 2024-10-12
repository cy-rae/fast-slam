﻿class EvaluationResults:
    """
    Class to store the results of the evaluation of the FastSLAM 2.0 algorithm.
    """
    def __init__(
            self,
            timestamp: str,
            average_deviation: float,
            x_deviation: float,
            y_deviation: float,
            angular_deviation: float,
            distance: float
    ):
        """
        Initialize the object with the passed parameters.
        """
        self.timestamp = timestamp
        self.average_deviation = average_deviation
        self.x_deviation = x_deviation
        self.y_deviation = y_deviation
        self.angular_deviation = angular_deviation
        self.distance = distance

    def to_dict(self) -> dict[str, float or str]:
        """
        Convert the evaluation results to a dictionary.
        :return: Returns the evaluation results as a dictionary
        """
        return {
            "timestamp": self.timestamp,
            "average_deviation": self.average_deviation,
            "x_deviation": self.x_deviation,
            "y_deviation": self.y_deviation,
            "angular_deviation": self.angular_deviation,
            "distance": self.distance
        }