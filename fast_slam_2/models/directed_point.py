from fast_slam_2.models.point import Point


class DirectedPoint(Point):
    """
    Class to represent a directed point in 2D space with a yaw value / angle in radians.
    """

    def __init__(self, x: float, y: float, yaw: float):
        super().__init__(x, y)
        self.yaw = yaw

    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'yaw': self.yaw,
        }
