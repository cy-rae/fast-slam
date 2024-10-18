import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


class MapUtils:
    """
    Utils class to plot the map with the estimated robot pose, actual robot pose, particles and landmarks plot the
    evaluation results.
    """

    @staticmethod
    def plot_map(
            estimated_robot_pos: tuple[float, float, float],
            actual_robot_pos: tuple[float, float, float],
            particles: list[tuple[float, float, float]],
            landmarks: list[tuple[float, float]],
            results: dict[str, float or str]
    ):
        """
        Plot the map with the estimated robot pose, actual robot pose, particles, landmarks and evaluation results.
        :param actual_robot_pos: The actual robot position represented as a tuple (x, y, yaw)
        :param estimated_robot_pos: The estimated robot position represented as a tuple (x, y, yaw)
        :param particles: The particles represented as a list of tuples (x, y, yaw)
        :param landmarks: The landmarks represented as a list of tuples (x, y)
        :param results: The evaluation results represented as a dictionary
        """
        try:
            # Initialize the plot
            fig, ax = MapUtils.__init_plot()

            # Plot the estimated robot position as a red arrow
            MapUtils.__plot_as_arrows(
                ax,
                directed_points=[estimated_robot_pos],
                scale=5,
                color='red',
                zorder=4,
                label='Estimated robot position'
            )

            # Plot the actual robot position as a black arrow
            MapUtils.__plot_as_arrows(
                ax,
                directed_points=[actual_robot_pos],
                scale=5,
                color='black',
                zorder=3,
                label='Actual robot position'
            )

            # Plot the particles as blue arrows
            MapUtils.__plot_as_arrows(
                ax,
                directed_points=particles,
                scale=5,
                color='blue',
                zorder=2,
                label='Particles'
            )

            # Mark landmarks as green dots
            MapUtils.__plot_as_dots(
                ax,
                landmarks,
                'g',
                zorder=1,
                label='Landmarks'
            )

            # Add the legend below the plot
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1)

            # Add the evaluation results as text under the plot
            MapUtils.__add_results_text(fig, results)

            # Show the plot
            plt.show()
        except Exception as e:
            print(e)

    @staticmethod
    def __init_plot() -> tuple[plt.Figure, Axes]:
        """
        Initialize the plot
        """
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 12))

        # Set limits and labels
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.axhline(0, color='black', linewidth=1)  # x axis
        ax.axvline(0, color='black', linewidth=1)  # y axis

        # Axis labels
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.text(-11, 11, "Map created by the FastSLAM 2.0 algorithm", fontsize=12, color='black')
        ax.grid(True)  # Show grid

        return fig, ax

    @staticmethod
    def __plot_as_arrows(
            ax: Axes,
            directed_points: list[tuple[float, float, float]],
            scale: float,
            color: str,
            zorder: int,
            label: str or None
    ):
        """
        Plot the passed directed points as arrows with the passed scale and color.
        :param ax: The axis to plot the arrows on
        :param directed_points: This list contains all the directed points which will be represented as arrows
        :param scale: The scale of the arrow
        :param color: The color of the arrow
        :param zorder: The zorder of the arrow that determines the order of the elements in the plot
        :param label: The label of the arrow that will be shown in the legend
        """
        for i, (x, y, yaw) in enumerate(directed_points):
            # Calculate the vector components
            dx = np.cos(yaw)  # x-component
            dy = np.sin(yaw)  # y-component

            # Draw the arrow. Add label only to the first arrow to avoid duplicate entries in the legend
            if i == 0 and label:
                ax.quiver(
                    x, y, dx, dy,
                    angles='xy', scale_units='inches', scale=scale, color=color, zorder=zorder, label=label
                )
            else:
                ax.quiver(
                    x, y, dx, dy,
                    angles='xy', scale_units='inches', scale=scale, color=color, zorder=zorder
                )

    @staticmethod
    def __plot_as_dots(
            ax: Axes,
            points: list[tuple[float, float]],
            color: str,
            zorder: int,
            label: str
    ):
        """
        Plot the passed points as dots. The color of the dots is determined by the passed color parameter.
        :param ax: The axis to plot the dots on
        :param points: This list contains all the points which will be represented as dots in the map
        :param color: The color of the dot ('g' -> green)
        :param zorder: The zorder of the dot that determines the order of the elements in the plot
        :param label: The label of the dot that will be shown in the legend
        """
        if label:
            ax.plot(
                [p[0] for p in points], [p[1] for p in points],
                color + 'o', markersize=5, zorder=zorder, label=label
            )
        else:
            ax.plot(
                [p[0] for p in points], [p[1] for p in points],
                color + 'o', markersize=5, zorder=zorder
            )

    @staticmethod
    def __add_results_text(fig: plt.Figure, results: dict):
        """
        Add the results as text under the plot.
        :param fig: The figure to add the text to
        :param results: The results to add as text
        """
        # Adjust the layout to add space at the bottom
        fig.subplots_adjust(bottom=0.25)  # Adjust the bottom to make space for the text

        fig.text(
            0.1,
            0.02,
            f"Timestamp: {results['timestamp']}\n"
            f"∅ deviation: {results['average_deviation']}%\n"
            f"X-deviation: {results['x_deviation']}%\n"
            f"Y-deviation: {results['y_deviation']}%\n"
            f"Angular deviation: {results['angular_deviation']}%\n"
            f"Distance between the actual and estimated robot position: {results['distance']}m",
            ha='left',
            fontsize=12
        )
