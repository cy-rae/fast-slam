import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RANSACRegressor

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


class LandmarkService:
    @staticmethod
    def get_measurements_to_landmarks(filtered_points: np.ndarray) -> tuple[list, list]:
        """
        Extract landmarks from the scanned points using the IEPF algorithm.
        :param scanned_points: The scanned points
        :return: Returns a list with the extracted landmarks
        """
        line_segments: list[tuple[tuple[float, float], tuple[float, float]]] = LandmarkService.__get_line_segments(
            filtered_points)

        print('\nLine segments', line_segments)

        # Plot lines
        plt.figure(figsize=(8, 6))
        plt.scatter(log[:, 0], log[:, 1], color='blue', s=10, label='Originalpunkte')
        plt.title('Originale Punkte')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        for line in line_segments:
            plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], color='red', linewidth=1)
        plt.show()

        corners: list[tuple[float, float]] = LandmarkService.__find_corners(line_segments)

        print('Corners', corners)


        # Plot corners
        plt.figure(figsize=(8, 6))
        plt.scatter(log[:, 0], log[:, 1], color='blue', s=10, label='Originalpunkte')
        plt.title('Originale Punkte')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        for corner in corners:
            plt.scatter(corner[0], corner[1], color='red', s=50)
        plt.show()

    @staticmethod
    def __get_line_segments(
            scanned_points: np.ndarray,
            max_lines=1,
            min_samples=25,
            residual_threshold=0.1
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """
        Extract line segments from the scanned points using the RANSAC algorithm.
        :param scanned_points: The scanned points to extract line segments from
        :param max_lines: The maximum number of lines to extract
        :param min_samples: The minimum number of samples needed to fit a line
        :param residual_threshold: The residual threshold to determine how close a point has to be to the line to be considered an inlier
        :return: Returns a list of line segments represented a tuples of endpoints of the line
        """
        x, y = scanned_points[:, 0], scanned_points[:, 1]
        endpoints: list[tuple[tuple[float, float], tuple[float, float]]] = []
        remaining_points = np.column_stack((x, y))

        for _ in range(max_lines):
            # Convert the remaining points to a 2D array
            X: np.ndarray = remaining_points[:, 0].reshape(-1, 1)
            Y: np.ndarray = remaining_points[:, 1].reshape(-1, 1)

            # Fit a line to the scanned points using RANSAC.
            # The residual threshold determines how close a point has to be to the line to be considered an inlier.
            # The min samples determine the minimum number of points that are needed to fit a line.
            ransac = RANSACRegressor(min_samples=min_samples, residual_threshold=residual_threshold)
            ransac.fit(X, Y)

            # Extract the inliers which are the points that are close to the line and the outliers
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)

            # If there are no inliers, break the loop
            if not np.any(inlier_mask):
                break

            # Get the corner points of the line (most left and most right point)
            inliers_x = X[inlier_mask]
            inliers_y = Y[inlier_mask]
            endpoint1: tuple[float, float] = (
                float(inliers_x.min()),
                float(inliers_y[inliers_x.argmin()][0])
            )
            endpoint2: tuple[float, float] = (
                float(inliers_x.max()),
                float(inliers_y[inliers_x.argmax()][0])
            )
            endpoints.append((endpoint1, endpoint2))

            # Remove the inliers from the remaining points
            remaining_points = remaining_points[outlier_mask]

            #plot inlier points
            plt.figure(figsize=(8, 6))
            plt.scatter(X[inlier_mask], Y[inlier_mask], color='red', s=10, label='Inliers')
            plt.title('Inliers')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)
            plt.legend()
            plt.show()


            # Plot remaining points
            plt.figure(figsize=(8, 6))
            plt.scatter(remaining_points[:, 0], remaining_points[:, 1], color='blue', s=10, label='Remaining')
            plt.title('Remaining Points')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)
            plt.legend()
            plt.show()

            # Break the loop if there are not enough remaining points
            if len(remaining_points) < min_samples:
                break

        return endpoints

    @staticmethod
    def __find_corners(
            line_segments: list[tuple[tuple[float, float], tuple[float, float]]],
            threshold=0.15
    ) -> list[tuple[float, float]]:
        """
        Find corners by comparing the Euclidean distance between the endpoints of the line segments.
        :param line_segments: List of line segments represented as tuples of endpoints
        :param threshold: The threshold to determine how close two points have to be, to be considered a corner
        :return: Returns a list of corners represented as tuples of x and y coordinates
        """
        corners: list[tuple[float, float]] = []  # List of corners
        used_points = set()  # Previously used points to prevent duplicates

        # Function to calculate the Euclidean distance between two points
        def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]):
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        # Loop through all combinations of line segments
        for i in range(len(line_segments)):
            for j in range(i + 1, len(line_segments)):
                # Calculate the angle between the two lines in degrees
                angle: float = LandmarkService.angle_between_lines(line_segments[i], line_segments[j])

                # If the angle is smaller than 45 degrees the corner will be ignored
                if angle < 45:
                    break

                # Compare all combinations of points
                for p1 in line_segments[i]:
                    for p2 in line_segments[j]:
                        # Calculate the Euclidean distance between the points
                        if euclidean_distance(p1, p2) < threshold:
                            # Add the points to the list of corners if they are close to each other
                            if p1 not in used_points:
                                corners.append(p1)
                                used_points.add(p1)

        return corners

    @staticmethod
    def angle_between_lines(
            line1: tuple[tuple[float, float], tuple[float, float]],
            line2: tuple[tuple[float, float], tuple[float, float]]
    ) -> float:
        # Get the endpoints of the lines
        p1, p2 = line1
        p3, p4 = line2

        # Calculate the direction vectors of the lines
        d1: tuple[float, float] = (p2[0] - p1[0], p2[1] - p1[1])
        d2: tuple[float, float] = (p4[0] - p3[0], p4[1] - p3[1])

        # Calculate the dot product of the direction vectors
        dot_product: float = d1[0] * d2[0] + d1[1] * d2[1]

        # Calculate the norm of the direction vectors
        norm_d1: float = math.sqrt(d1[0] ** 2 + d1[1] ** 2)
        norm_d2: float = math.sqrt(d2[0] ** 2 + d2[1] ** 2)

        # Prevent division by zero
        if norm_d1 == 0 or norm_d2 == 0:
            return 0

        # Calculate the cosine of the angle between the lines
        cos_theta: float = dot_product / (norm_d1 * norm_d2)

        # Clamp the cosine value to the range [-1, 1]
        cos_theta = max(-1, min(1, cos_theta))

        # Calculate the angle between the lines in degrees
        angle: float = math.acos(cos_theta)
        angle_degrees: float = math.degrees(angle)

        return angle_degrees

    @staticmethod
    def __calculate_distance_and_angle(x: float, y: float):
        """
        Calculate the distance and angle of a point to the origin (0, 0). The angle is rotated by -90 degrees.
        :param x: The x coordinate of the point
        :param y: The y coordinate of the point
        :return: Returns the distance(s) and angle(s) of the point(s) to the origin (0, 0)
        """
        distance = math.sqrt(x ** 2 + y ** 2)
        angle = math.atan2(y, x)
        return distance, angle

LandmarkService.get_measurements_to_landmarks(log)
