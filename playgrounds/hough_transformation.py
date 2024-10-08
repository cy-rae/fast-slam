import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from sklearn.cluster import DBSCAN

class LandmarkService:
    __padding: int = 20
    __scale_factor: int = 100

    @staticmethod
    def get_landmarks(scanned_points: np.ndarray):
        """
        Extract landmarks from the given scanned points using hough transformation and DBSCAN clustering.
        :param scanned_points: Scanned points in the form of a numpy array
        :return: Returns the extracted landmarks
        """
        # Create hough transformation image
        image, width, height = LandmarkService.__create_hough_transformation_image(scanned_points)

        # Detect lines using hough transformation
        lines = LandmarkService.__hough_line_detection(image)

        # Calculate the intersection points and cluster them to prevent multiple points for the same intersection
        # which can happen when multiple lines were detected for the same edge
        intersection_points = LandmarkService.__calculate_intersections(lines, width, height)
        intersection_points = LandmarkService.__convert_back_to_original_space(scanned_points, intersection_points)
        intersection_points = LandmarkService.__cluster_points(intersection_points, 0.5, 1)

        # Convert the intersection points back to the original coordinate space
        return intersection_points

    @staticmethod
    def __create_hough_transformation_image(scanned_points: np.ndarray):
        # Get the scaled min and max values of the scanned points
        min_x = int(np.min(scanned_points[:, 0] * LandmarkService.__scale_factor))
        min_y = int(np.min(scanned_points[:, 1] * LandmarkService.__scale_factor))
        max_x = int(np.max(scanned_points[:, 0] * LandmarkService.__scale_factor))
        max_y = int(np.max(scanned_points[:, 1] * LandmarkService.__scale_factor))

        # Calculate the offset to bring all points into the positive coordinate system for the transformation
        offset_x = -min_x if min_x < 0 else 0
        offset_y = -min_y if min_y < 0 else 0
        offset_x += LandmarkService.__padding  # Apply padding to avoid drawing points at the edge of the image
        offset_y += LandmarkService.__padding

        # Create a new image for the transformation with the offsets
        width = max_x + offset_x + LandmarkService.__padding
        height = max_y + offset_y + LandmarkService.__padding
        image = np.zeros((height, width), dtype=np.uint8)

        # Scale and add the scanned points to the image as circles
        for point in scanned_points:
            x = int(point[0] * LandmarkService.__scale_factor) + offset_x
            y = int(point[1] * LandmarkService.__scale_factor) + offset_y
            cv2.circle(image, center=(x, y), radius=2, color=255, thickness=-1)

        return image, width, height

    @staticmethod
    def __hough_line_detection(image):
        """
        Detect lines in the given image using the hough transformation.
        :param image: The image to detect lines in
        :return: Returns the detected lines
        """
        # Schritt 4: Kantenextraktion mit Canny
        edges = cv2.Canny(image, 100, 150, apertureSize=3)

        # Schritt 5: Verwende die Hough-Transformation zur Linienerkennung
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 90)

        return lines

    @staticmethod
    def __calculate_intersections(lines, width, height) -> list[tuple[float, float]]:
        """
        Calculate the intersection points of the given lines.
        :param lines: The lines to calculate the intersection points for
        :param width: The width of the image
        :param height: The height of the image
        :return: Returns the intersection points
        """
        # Check if no lines were detected
        if lines is None:
            return []

        # Calculate the intersection points of the lines
        intersections: list[tuple[float, float]] = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                # Get the rho and theta values of the lines
                rho1, theta1 = lines[i][0]
                rho2, theta2 = lines[j][0]

                # Calculate the angle difference between the lines
                angle_diff: float = abs(theta1 - theta2)
                angle_diff: float = min(angle_diff, np.pi - angle_diff)  # Normalize the angle difference to [0, pi]

                # If the angle difference is too small, the lines are almost parallel and the intersection point will be ignored
                if angle_diff < np.deg2rad(45):
                    continue

                # Calculate the coefficients of the lines
                a1, b1 = np.cos(theta1), np.sin(theta1)
                a2, b2 = np.cos(theta2), np.sin(theta2)

                # Calculate the determinant of the lines to check if they intersect
                determinant: float = a1 * b2 - a2 * b1
                if abs(determinant) > 1e-10:
                    # Calculate the intersection point
                    x: float = (b2 * rho1 - b1 * rho2) / determinant
                    y: float = (a1 * rho2 - a2 * rho1) / determinant

                    # Only consider intersection points within the image bounds
                    if 0 <= x < width and 0 <= y < height:
                        intersections.append((x, y))

        return intersections

    @staticmethod
    def __cluster_points(point_list: list[tuple[float, float]], eps=10, min_samples=1) -> list[tuple[float, float]]:
        """
        Cluster the given points using DBSCAN.
        :param point_list: The points to cluster
        :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other
        :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point
        :return: Returns the clustered points
        """
        # Convert the points to a numpy array
        points: ndarray = np.array(point_list)

        # Use DBSCAN to cluster the points
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

        # Extract the unique cluster labels
        labels = db.labels_
        unique_labels = set(labels)

        # Iterate through the unique clusters and collect their centroids
        cluster_centers: list[tuple[float, float]] = []
        for label in unique_labels:
            # -1 is the label for noise which can be ignored
            if label == -1:
                continue

            # Get the points which belong to the current cluster
            cluster_points: ndarray = points[labels == label]

            # Calculate centroids
            centroids: tuple[float, float] = cluster_points.mean(axis=0)
            cluster_centers.append(centroids)

        return cluster_centers

    @staticmethod
    def __convert_back_to_original_space(scanned_points, cluster_centers):
        """
        Convert the clustered points back to the original coordinate space.
        :param scanned_points: The scanned points
        :param cluster_centers: The clustered points
        :return:
        """
        original_points: list[tuple[float, float]] = []

        # Calculate the offset to move all points into the correct position of the coordinate system
        min_x = int(np.min(scanned_points[:, 0] * LandmarkService.__scale_factor))
        min_y = int(np.min(scanned_points[:, 1] * LandmarkService.__scale_factor))
        offset_x = -min_x if min_x < 0 else 0
        offset_y = -min_y if min_y < 0 else 0
        offset_x += LandmarkService.__padding
        offset_y += LandmarkService.__padding

        # Calculate the original points
        for x, y in cluster_centers:
            original_x = (x - offset_x) / LandmarkService.__scale_factor
            original_y = (y - offset_y) / LandmarkService.__scale_factor
            original_points.append((original_x, original_y))

        return original_points

log = np.array([
    [0.00602347, - 0.79963241],
    [0.01514805, - 0.8055237],
    [0.02867247, - 0.81425712],
    [0.04335421, - 0.82374008],
    [0.05847621, - 0.83350987],
    [0.07400203, - 0.84354307],
    [0.08995608, - 0.85385552],
    [0.10636664, - 0.86446554],
    [0.12326419, - 0.87539291],
    [0.14068167, - 0.88665902],
    [0.15865475, - 0.89828702],
    [0.17722213, - 0.91030203],
    [0.19642586, - 0.92273134],
    [0.21631174, - 0.93560467],
    [0.23692975, - 0.94895447],
    [0.25833457, - 0.96281623],
    [0.28058614, - 0.97722886],
    [0.30375034, - 0.99223516],
    [0.32789974, - 1.0078822],
    [0.35311449, - 1.02422202],
    [0.37948335, - 1.04131226],
    [0.40710491, - 1.05921695],
    [0.43608898, - 1.07800738],
    [0.46655826, - 1.0977632],
    [0.49865031, - 1.11857372],
    [0.53251986, - 1.14053942],
    [0.56834166, - 1.16377377],
    [0.60631379, - 1.18840541],
    [0.6466617, - 1.21458072],
    [0.68964316, - 1.24246706],
    [0.73555427, - 1.27225671],
    [0.78473693, - 1.30417168],
    [0.83758809, - 1.3384697],
    [0.89456462, - 1.37544299],
    [0.9559953, - 1.41515939],
    [1.01997982, - 1.45468515],
    [1.07695709, - 1.48111917],
    [1.11592094, - 1.48066107],
    [1.14035883, - 1.45980978],
    [1.15965756, - 1.43234241],
    [1.17797014, - 1.4041184],
    [1.19603604, - 1.37612753],
    [1.21392215, - 1.34842091],
    [1.23164357, - 1.32097939],
    [1.24921344, - 1.29378248],
    [1.26664453, - 1.26681038],
    [1.28394922, - 1.2400439],
    [1.30113951, - 1.21346433],
    [1.31822713, - 1.18705348],
    [1.33522354, - 1.16079358],
    [1.35213996, - 1.13466728],
    [1.36898738, - 1.10865749],
    [1.38577661, - 1.08274748],
    [1.40251836, - 1.05692075],
    [1.41922319, - 1.03116101],
    [1.43590152, - 1.00545209],
    [1.45256372, - 0.97977797],
    [1.46922013, - 0.95412273],
    [1.48588111, - 0.92847048],
    [1.50255696, - 0.90280533],
    [1.51925801, - 0.87711135],
    [1.5359947, - 0.85137255],
    [1.55277751, - 0.82557281],
    [1.56961702, - 0.79969585],
    [1.586524, - 0.77372523],
    [1.60350935, - 0.74764423],
    [1.62058416, - 0.72143587],
    [1.63775972, - 0.69508283],
    [1.6550476, - 0.66856742],
    [1.67245967, - 0.64187154],
    [1.69000809, - 0.61497659],
    [1.70770538, - 0.58786347],
    [1.72556444, - 0.56051247],
    [1.74359864, - 0.53290325],
    [1.76182177, - 0.50501474],
    [1.78024818, - 0.47682513],
    [1.7988872, - 0.44831073],
    [1.81756895, - 0.41941555],
    [1.83404744, - 0.38972639],
    [1.83826829, - 0.3575692],
    [1.81542213, - 0.32089447],
    [1.76606788, - 0.28082807],
    [1.70691232, - 0.24102014],
    [1.64933065, - 0.20356223],
    [1.59554145, - 0.16866662],
    [1.54533364, - 0.13609471],
    [1.49833829, - 0.1056044],
    [1.45422393, - 0.0769849],
    [1.41263899, - 0.05005288],
    [1.37312988, - 0.02464916],
    [1.33554473e+00, - 6.37339858e-04],
    [1.30018419, 0.02210368],
    [1.26690734, 0.04368899],
    [1.23533229, 0.06422159],
    [1.20525084, 0.0837898],
    [1.17653644, 0.1024715],
    [1.14908139, 0.12033651],
    [1.12278891, 0.13744768],
    [1.09757157, 0.15386177],
    [1.07335027, 0.16963016],
    [1.05005321, 0.1847995],
    [1.02761507, 0.19941225],
    [1.00597622, 0.21350713],
    [0.98508215, 0.22711952],
    [0.96488292, 0.24028186],
    [0.94533274, 0.25302393],
    [0.92638941, 0.26537313],
    [0.90801407, 0.27735473],
    [0.89017079, 0.28899205],
    [0.87282636, 0.30030669],
    [0.85594997, 0.31131866],
    [0.83951301, 0.32204656],
    [0.82348886, 0.3325077],
    [0.80785273, 0.34271819],
    [0.79258145, 0.35269311],
    [0.77765337, 0.36244652],
    [0.76304819, 0.37199163],
    [0.74874691, 0.38134081],
    [0.73473163, 0.39050571],
    [0.72098551, 0.39949728],
    [0.70749267, 0.40832585],
    [0.69423811, 0.41700118],
    [0.68120767, 0.42553251],
    [0.6683879, 0.43392859],
    [0.65576603, 0.44219774],
    [0.64332994, 0.45034785],
    [0.63106808, 0.45838648],
    [0.61896943, 0.4663208],
    [0.60702346, 0.47415768],
    [0.59522006, 0.48190373],
    [0.58354958, 0.48956524],
    [0.57200269, 0.49714831],
    [0.56057045, 0.50465879],
    [0.54924421, 0.51210232],
    [0.53801561, 0.51948437],
    [0.52687654, 0.52681025],
    [0.51581917, 0.53408511],
    [0.50483584, 0.54131395],
    [0.4939191, 0.54850169],
    [0.48306166, 0.55565309],
    [0.47225637, 0.56277284],
    [0.46149623, 0.56986554],
    [0.45077434, 0.57693573],
    [0.44008391, 0.58398788],
    [0.42941821, 0.59102643],
    [0.41877058, 0.59805578],
    [0.40813439, 0.60508028],
    [0.39750306, 0.61210429],
    [0.38686999, 0.61913216],
    [0.3762286, 0.62616825],
    [0.36557228, 0.63321693],
    [0.35489437, 0.6402826],
    [0.34418818, 0.64736972],
    [0.33344692, 0.65448277],
    [0.32266372, 0.6616263],
    [0.31183161, 0.66880495],
    [0.3009435, 0.67602343],
    [0.28999214, 0.68328658],
    [0.27897013, 0.69059933],
    [0.26786987, 0.69796672],
    [0.25668355, 0.70539394],
    [0.24540316, 0.71288635],
    [0.23402041, 0.72044948],
    [0.22252675, 0.72808904],
    [0.21091329, 0.73581096],
    [0.19917083, 0.74362137],
    [0.1872898, 0.75152669],
    [0.17526024, 0.75953358],
    [0.16307171, 0.76764903],
    [0.15071334, 0.7758803],
    [0.13817371, 0.78423502],
    [0.12544085, 0.79272123],
    [0.11250217, 0.80134735],
    [0.09934442, 0.81012229],
    [0.0859536, 0.8190554],
    [0.07231493, 0.8281566],
    [0.0584148, 0.83743502],
    [0.04430362, 0.84685699],
    [0.03077833, 0.85589002],
    [0.02141364, 0.86214535]
])

# Beispielaufruf
if __name__ == "__main__":
    corners = LandmarkService.get_landmarks(log)

    # Schritt 1: Visualisiere die originalen Punkte
    plt.figure(figsize=(8, 6))
    plt.scatter(log[:, 0], log[:, 1], color='blue', s=10, label='Originalpunkte')
    plt.scatter(np.array(corners)[:, 0], np.array(corners)[:, 1], color='red', s=50, label='Eckpunkte')
    plt.title('Originale Punkte')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.gca().invert_yaxis()  # Invertiere die Y-Achse, um das Bild wie im OpenCV-Bild zu orientieren
    plt.show()