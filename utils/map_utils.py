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
            ax = MapUtils.__init_plot()

            MapUtils.__plot_as_arrows(ax, directed_points=[robot], scale=5,
                                      color='red', zorder=2)  # Plot the robot as a red arrow
            MapUtils.__plot_as_arrows(ax, directed_points=particles, scale=6,
                                      color='blue', zorder=1)  # Plot the particles as blue arrows
            MapUtils.__plot_as_dots(ax, landmarks, 'g')  # Mark landmarks as green dots

            # Show the plot
            plt.show()
        except Exception as e:
            print(e)

    @staticmethod
    def __init_plot():
        """
        Initialize the plot
        """
        # Create a figure and axis
        _, ax = plt.subplots(figsize=(10, 10))

        # Set limits and labels
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.axhline(0, color='black', linewidth=1)  # x axis
        ax.axvline(0, color='black', linewidth=1)  # y axis

        # Axis labels
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.text(-11, 11, "Map created by the fast_slam_2 2.0 algorithm", fontsize=12, color='black')
        ax.grid(True) # Show grid

        return ax

    @staticmethod
    def __plot_as_arrows(ax, directed_points: list[tuple[float, float, float]], scale: float, color: str, zorder: int):
        """
        Plot the passed directed points as arrows with the passed scale and color.
        :param directed_points: This list contains all the directed points which will be represented as arrows
        :param scale: The scale of the arrow
        :param color: The color of the arrow
        """
        for (x, y, yaw) in directed_points:
            # Calculate the vector components
            dx = np.cos(yaw)  # x-component
            dy = np.sin(yaw)  # y-component

            # Draw the arrow
            ax.quiver(x, y, dx, dy, angles='xy', scale_units='inches', scale=scale, color=color, zorder=zorder)

    @staticmethod
    def __plot_as_dots(ax, points: list[tuple[float, float]], color: str):
        """
        Plot the passed points as dots. The color of the dots is determined by the passed color parameter.
        :param points: This list contains all the points which will be represented as dots in the map
        :param color: The color of the dot ('k' -> black, 'g' -> green)
        """
        for point in points:
            ax.plot(point[0], point[1], color + 'o', markersize=5, label='Punkte')
