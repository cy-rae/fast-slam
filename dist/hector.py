import numpy as np


# region Models
class Point:
    """
    Class to represent a point in 2D space.
    """

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def as_vector(self):
        """
        Get the pose/mean of the point as a vector [x, y].
        :return: Returns the position of the point as a numpy array [x, y]
        """
        return np.array([self.x, self.y])


class DirectedPoint(Point):
    """
    Class to represent a point in 2D space with a yaw value / angle in degrees.
    """

    def __init__(self, x: float, y: float, yaw: float):
        super().__init__(x, y)
        self.yaw = yaw


class Robot(DirectedPoint):
    """
    This class represents the robot
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        super().__init__(x, y, yaw)
        self.x_prev = 0.0
        self.y_prev = 0.0
        self.yaw_prev = 0.0
        self.timestamp_prev = 0.0
        self.x_prev = 0.0
        self.y_prev = 0.0
        self.yaw_prev = 0.0

    @staticmethod
    def scan_environment() -> list[Point]:
        """
        Scan the environment using the laser data and return a list of points that were scanned by the laser.
        :return: Return a list of points that were scanned by the laser
        """
        # Get laser data from the robot. Laser data contains the distances and angles to obstacles in the environment.
        laser_data = HAL.getLaserData()

        # Convert each laser data value to a point
        scanned_points: list[Point] = []
        for i in range(180):  # Laser data has 180 values
            # Extract the distance at index i
            dist = laser_data.values[i]

            # Skip invalid distances (e.g., min or max range)
            if dist < laser_data.minRange or dist > laser_data.maxRange:
                continue

            # The final angle is centered (zeroed) at the front of the robot.
            angle = np.radians(i - 90)

            # Compute x, y coordinates from distance and angle
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)
            scanned_points.append(Point(x, y))
        return scanned_points


# endregion

def polar_to_cartesian(ranges, angles, pose):
    """
    Konvertiert Polarkoordinaten in kartesische Koordinaten im globalen Rahmen.
    """
    x, y, theta = pose
    xs = ranges * np.cos(angles + theta) + x
    ys = ranges * np.sin(angles + theta) + y
    return np.vstack((xs, ys)).T


# Beispielhafte Laserdaten als Ersatz für die Eingabe von deinem Roboter
class LaserData:
    def __init__(self, values):
        self.values = values


def update_map(occupancy_map, scan_points):
    """
    Zeichnet die Scanpunkte in die Occupancy Map.
    """
    for point in scan_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < occupancy_map.shape[0] and 0 <= y < occupancy_map.shape[1]:
            occupancy_map[x, y] = 1  # Punkt wird als besetzt markiert


def simple_scan_matching(occupancy_map, scan_points, initial_pose):
    """
    Ein einfacher Scan-Matching-Algorithmus, der versucht, die Fehler zu minimieren.
    """
    best_pose = initial_pose
    min_error = float('inf')

    # Suche durch verschiedene Posen
    for dx in np.linspace(-1, 1, 5):
        for dy in np.linspace(-1, 1, 5):
            for dtheta in np.linspace(-0.1, 0.1, 5):
                new_pose = initial_pose + np.array([dx, dy, dtheta])
                transformed_points = polar_to_cartesian(laser_data.values, angles, new_pose)

                # Fehlerberechnung (Anzahl der Punkte, die auf besetzten Zellen landen)
                error = 0
                for point in transformed_points:
                    x, y = int(point[0]), int(point[1])
                    if 0 <= x < occupancy_map.shape[0] and 0 <= y < occupancy_map.shape[1]:
                        error += occupancy_map[x, y]

                if error < min_error:
                    min_error = error
                    best_pose = new_pose

    return best_pose


robot = Robot()

# Initialisiere die Karte (Occupancy Grid Map)
map_size = (1000, 1000)
occupancy_map = np.zeros(map_size)

# Winkel, die den Scan-Positionen entsprechen
angles = np.linspace(-np.pi / 2, np.pi / 2, 180)  # von -90° bis 90°

# Parameter
map_resolution = 1  # jeder Kartenwert entspricht 1 Einheit

# Polar in kartesische Koordinaten umwandeln
scan_points = polar_to_cartesian(laser_data.values, angles, robot)

# Optimierte Pose berechnen
pose = simple_scan_matching(occupancy_map, scan_points, robot)

# Aktualisiere die Karte mit den aktuellen Scandaten
update_map(occupancy_map, scan_points)
