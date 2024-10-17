from fast_slam_2.models.point import Point


class DirectedPoint(Point):
    """
    Class to represent a directed point in 2D space with a yaw value / angle in radians.
    """

    def __init__(self, x: float, y: float, yaw: float):
        """
        Initialize the directed point with the passed parameters.
        :param x: The x coordinate of the directed point
        :param y: The y coordinate of the directed point
        :param yaw: The angle of the directed point in radians
        """
        super().__init__(x, y)
        self.yaw = yaw

    def to_dict(self):
        """
        Convert the directed point to a dictionary.
        :return: Returns the dictionary representation of the directed point.
        """
        return {
            'x': self.x,
            'y': self.y,
            'yaw': self.yaw,
        }
