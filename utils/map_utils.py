import numpy as np
import matplotlib.pyplot as plt

from FastSLAM2.models.directed_point import DirectedPoint
from FastSLAM2.models.landmark import Landmark
from FastSLAM2.models.particle import Particle
from FastSLAM2.models.point import Point
from FastSLAM2.models.robot import Robot


class MapUtils:
    """
    Utils class to plot the map with the robot, particles, landmarks and obstacles/borders.
    """

    @staticmethod
    def plot_map(robot: DirectedPoint, particles: list[DirectedPoint], landmarks: list[Point]):
        """
        Plot the map with the robot, particles, landmarks and obstacles/borders.
        """
        try:
            fig, ax = MapUtils.__init_plot()
            MapUtils.__plot_as_arrows(ax, directed_points=[robot], scale=10, color='red')  # Plot the robot as a red arrow
            MapUtils.__plot_as_arrows(ax, directed_points=particles, scale=7, color='blue')  # Plot the particles as blue arrows
            MapUtils.__plot_as_dots(ax, landmarks, 'green')  # Mark landmarks as green dots

            # Show the plot
            plt.show(fig)
        except Exception as e:
            print(e)

    @staticmethod
    def __init_plot():
        """
        Initialize the plot
        """
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))

        # Set limits and labels
        ax.set_xlim(0, 1500)
        ax.set_ylim(0, 1500)
        ax.axhline(750, color='black', linewidth=2)  # X-Achse
        ax.axvline(750, color='black', linewidth=2)  # Y-Achse

        # Axis labels
        ax.text(1400, 760, "X-axis", fontsize=12, color='black')
        ax.text(760, 10, "Y-axis", fontsize=12, color='black')
        ax.text(0, 10, "Map created by the FastSLAM2 2.0 algorithm", fontsize=12, color='black')

        return fig, ax

    @staticmethod
    def __plot_as_arrows(ax, directed_points: list[DirectedPoint], scale: float, color: str):
        """
        Plot the passed directed points as arrows with the passed scale and color.
        :param directed_points: This list contains all the directed points which will be represented as arrows
        :param scale: The scale of the arrow
        :param color: The color of the arrow
        """
        center_x = 750  # Middle of the X-axis
        center_y = 750  # Middle of the Y-axis
        for obj in directed_points:
            # Calculate the start and end point of the arrow
            x_start = center_x + obj.x * 50  # Scale the X-coordinate
            y_start = center_y - obj.y * 50  # Scale the Y-coordinate
            x_end = x_start + np.cos(obj.yaw) * scale
            y_end = y_start - np.sin(obj.yaw) * scale

            # Draw the arrow
            ax.arrow(x_start, y_start, x_end - x_start, y_end - y_start,
                     head_width=5, head_length=10, fc=color, ec=color)

    @staticmethod
    def __plot_as_dots(ax, points: list[Point], color: str):
        """
        Plot the passed points as dots. The color of the dots is determined by the passed color parameter.
        :param points: This list contains all the points which will be represented as dots in the map
        :param color: The color of the dot ('k' -> black, 'g' -> green)
        """
        for point in points:
            x = 750 + point.x * 50  # Scale the X-coordinate
            y = 750 - point.y * 50  # Scale the Y-coordinate
            radius = 5  # Size of the dots
            ax.plot(x, y, 'o', color=color, markersize=radius)