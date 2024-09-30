import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.fast_slam_2 import DirectedPoint, Point


class Map:
    @staticmethod
    def plot_map(robot: DirectedPoint, particles: list[DirectedPoint], obstacles: list[Point], landmarks: list[Point]):
        """
        Plot the map with the robot, particles, landmarks and obstacles/borders.
        """
        try:
            image, draw = Map.__init_plot()
            Map.__plot_as_arrows(draw, directed_points=[robot], scale=5.5, color='red')  # Plot the robot as a red arrow
            Map.__plot_as_arrows(draw, directed_points=particles, scale=7, color='blue')  # Plot the particles as blue arrows
            Map.__plot_as_dots(draw, obstacles, 'black')  # Mark obstacles as black dots
            Map.__plot_as_dots(draw, landmarks, 'green')  # Mark landmarks as green dots

            # Save the plot as an image file
            image.save('/usr/share/nginx/html/images/map.jpg', 'JPEG')
        except Exception as e:
            print(e)

    @staticmethod
    def __init_plot():
        """
        Initialize the plot
        """
        # Bildgröße und Hintergrundfarbe
        width, height = 600, 600
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        # Achsen zeichnen
        center_x = width // 2
        center_y = height // 2
        draw.line((0, center_y, width, center_y), fill="black", width=2)  # X-Achse
        draw.line((center_x, 0, center_x, height), fill="black", width=2)  # Y-Achse

        # Achsenbeschriftungen
        font = ImageFont.load_default()
        draw.text((width - 100, center_y + 10), "X-axis", fill="black", font=font)
        draw.text((center_x + 10, 10), "Y-axis", fill="black", font=font)
        draw.text((width // 4, 10), "Map created by the FastSLAM 2.0 algorithm", fill="black", font=font)

        return image, draw

    @staticmethod
    def __plot_as_arrows(draw: ImageDraw.Draw, directed_points: list[DirectedPoint], scale: float, color: str):
        """
        Plot the passed directed points as arrows with the passed scale and color.
        :param directed_points: This list contains all the directed points which will be represented as arrows
        :param scale: The scale of the arrow
        :param color: The color of the arrow
        """
        center_x = 300  # Mittelpunkt der X-Achse
        center_y = 300  # Mittelpunkt der Y-Achse
        for obj in directed_points:
            # Berechnung der Endpunkte des Pfeils
            x_start = center_x + obj.x * 50  # Skaliere die X-Koordinate
            y_start = center_y - obj.y * 50  # Skaliere die Y-Koordinate
            x_end = x_start + np.cos(obj.get_yaw_rad()) * scale
            y_end = y_start - np.sin(obj.get_yaw_rad()) * scale
            # Zeichne den Pfeil
            draw.line((x_start, y_start, x_end, y_end), fill=color, width=3)
            # Zeichne die Pfeilspitze
            arrow_size = 5
            draw.line((x_end, y_end, x_end - arrow_size * np.cos(obj.get_yaw_rad() + np.pi / 6),
                        y_end + arrow_size * np.sin(obj.get_yaw_rad() + np.pi / 6)), fill=color, width=3)
            draw.line((x_end, y_end, x_end - arrow_size * np.cos(obj.get_yaw_rad() - np.pi / 6),
                        y_end + arrow_size * np.sin(obj.get_yaw_rad() - np.pi / 6)), fill=color, width=3)

    @staticmethod
    def __plot_as_dots(draw: ImageDraw.Draw, points: list[Point], color: str):
        """
        Plot the passed points as dots. The color of the dots is determined by the passed color parameter.
        :param points: This list contains all the points which will be represented as dots in the map
        :param color: The color of the dot ('k' -> black, 'g' -> green)
        """
        for point in points:
            x = 300 + point.x * 50  # Skaliere die X-Koordinate
            y = 300 - point.y * 50  # Skaliere die Y-Koordinate
            radius = 3
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
