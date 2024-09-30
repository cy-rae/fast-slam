import numpy as np
from matplotlib import pyplot as plt

from src.models.directed_point import DirectedPoint
from src.models.point import Point


class MapService:
    @staticmethod
    def plot_map(robot: DirectedPoint, particles: list[DirectedPoint], obstacles: list[Point], landmarks: list[Point]):
        """
        Plot the map with the robot, particles, landmarks and obstacles/borders.
        """
        try:
            MapService.__init_plot()
            MapService.__plot_as_arrows(directed_points=[robot], scale=5.5, color='r')  # Plot the robot as a red arrow
            MapService.__plot_as_arrows(directed_points=particles, scale=7,
                                        color='b')  # Plot the particles as blue arrows
            MapService.__plot_as_dots(obstacles, 'k')  # Mark obstacles as black dots
            MapService.__plot_as_dots(landmarks, 'g')  # Mark landmarks as green dots

            # Save the plot as an image file and plot it
            plt.savefig('/usr/share/nginx/html/images/map.jpg')
            plt.show()
        except Exception as e:
            print(e)

    @staticmethod
    def __init_plot():
        """
        Initialize the plot
        """
        plt.figure()
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Map created by the FastSLAM 2.0 algorithm')

    @staticmethod
    def __plot_as_arrows(directed_points: list[DirectedPoint], scale: float, color: str):
        """
        Plot the passed directed points as arrows with the passed scale and color.
        :param directed_points: This list contains all the directed points which will be represented as (short) arrows
        :param scale: The scale of the arrow
        :param color: The color of the arrow
        """
        plt.quiver(
            [obj.x for obj in directed_points],
            [particle.y for particle in directed_points],
            [np.cos(obj.get_yaw_rad()) for obj in directed_points],
            [np.sin(obj.get_yaw_rad()) for obj in directed_points],
            scale=scale,
            scale_units='inches',
            angles='xy',
            color=color
        )

    @staticmethod
    def __plot_as_dots(points: list[Point], color: str):
        """
        Plot the passed points as dots. The color of the dots is determined by the passed color parameter.
        :param points: This list contains all the points which will be represented as dots in the map
        :param color: The color of the dot ('k' -> black, 'g' -> green)
        """
        for landmark in points:
            plt.plot(landmark.x, landmark.y, color + 'o')  # 'o' -> circle marker
