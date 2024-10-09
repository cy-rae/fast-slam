import matplotlib.pyplot as plt
import numpy as np


class MapUtils:
    """
    Utils class to plot the map with the robot, particles, landmarks and obstacles/borders.
    """

    @staticmethod
    def plot_map(
            robot: tuple[float, float, float],
            particles: list[tuple[float, float, float]],
            landmarks: list[tuple[float, float]]
    ):
        """
        Plot the map with the robot, particles, landmarks and obstacles/borders.
        :param robot: The robot represented as a tuple (x, y, yaw)
        :param particles: The particles represented as a list of tuples (x, y, yaw)
        :param landmarks: The landmarks represented as a list of tuples (x, y)
        """
        try:
            fig, ax = MapUtils.__init_plot()
            MapUtils.__plot_as_arrows(ax, directed_points=[robot], scale=10,
                                      color='red')  # Plot the robot as a red arrow
            MapUtils.__plot_as_arrows(ax, directed_points=particles, scale=7,
                                      color='blue')  # Plot the particles as blue arrows
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
        ax.text(0, 10, "Map created by the fast_slam_2 2.0 algorithm", fontsize=12, color='black')

        return fig, ax

    @staticmethod
    def __plot_as_arrows(ax, directed_points: list[tuple[float, float, float]], scale: float, color: str):
        """
        Plot the passed directed points as arrows with the passed scale and color.
        :param directed_points: This list contains all the directed points which will be represented as arrows
        :param scale: The scale of the arrow
        :param color: The color of the arrow
        """
        center_x = 750  # Middle of the X-axis
        center_y = 750  # Middle of the Y-axis
        for directed_point in directed_points:
            # Calculate the start and end point of the arrow
            x_start = center_x + directed_point[0] * 50  # Scale the X-coordinate
            y_start = center_y - directed_point[1] * 50  # Scale the Y-coordinate
            x_end = x_start + np.cos(directed_point[2]) * scale
            y_end = y_start - np.sin(directed_point[2]) * scale

            # Draw the arrow
            ax.arrow(x_start, y_start, x_end - x_start, y_end - y_start,
                     head_width=5, head_length=10, fc=color, ec=color)

    @staticmethod
    def __plot_as_dots(ax, points: list[tuple[float, float]], color: str):
        """
        Plot the passed points as dots. The color of the dots is determined by the passed color parameter.
        :param points: This list contains all the points which will be represented as dots in the map
        :param color: The color of the dot ('k' -> black, 'g' -> green)
        """
        for point in points:
            x = 750 + point[0] * 50  # Scale the X-coordinate
            y = 750 - point[1] * 50  # Scale the Y-coordinate
            radius = 5  # Size of the dots
            ax.plot(x, y, 'o', color=color, markersize=radius)
