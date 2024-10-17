import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from sklearn.cluster import DBSCAN

class HoughTransformation:
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
        image, width, height = HoughTransformation.__create_hough_transformation_image(scanned_points)

        # Detect lines using hough transformation
        edges, lines = HoughTransformation.__hough_line_detection(image)

        # Plot Canny and Hough results
        HoughTransformation.__plot_canny_hough_results(image, edges, lines)

        # Calculate the intersection points and cluster them to prevent multiple points for the same intersection
        # which can happen when multiple lines were detected for the same edge
        intersection_points = HoughTransformation.__calculate_intersections(lines, width, height)
        intersection_points = HoughTransformation.__convert_back_to_original_space(scanned_points, intersection_points)
        intersection_points = HoughTransformation.__cluster_points(intersection_points, 0.5, 1)

        # Convert the intersection points back to the original coordinate space
        return intersection_points

    @staticmethod
    def __create_hough_transformation_image(scanned_points: np.ndarray):
        # Get the scaled min and max values of the scanned points
        min_x = int(np.min(scanned_points[:, 0] * HoughTransformation.__scale_factor))
        min_y = int(np.min(scanned_points[:, 1] * HoughTransformation.__scale_factor))
        max_x = int(np.max(scanned_points[:, 0] * HoughTransformation.__scale_factor))
        max_y = int(np.max(scanned_points[:, 1] * HoughTransformation.__scale_factor))

        # Calculate the offset to bring all points into the positive coordinate system for the transformation
        offset_x = -min_x if min_x < 0 else 0
        offset_y = -min_y if min_y < 0 else 0
        offset_x += HoughTransformation.__padding  # Apply padding to avoid drawing points at the edge of the image
        offset_y += HoughTransformation.__padding

        # Create a new image for the transformation with the offsets
        width = max_x + offset_x + HoughTransformation.__padding
        height = max_y + offset_y + HoughTransformation.__padding
        image = np.zeros((height, width), dtype=np.uint8)

        # Scale and add the scanned points to the image as circles
        for point in scanned_points:
            x = int(point[0] * HoughTransformation.__scale_factor) + offset_x
            y = int(point[1] * HoughTransformation.__scale_factor) + offset_y
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
        # edges = cv2.Canny(image, 100, 150, apertureSize=3)

        # Schritt 5: Verwende die Hough-Transformation zur Linienerkennung
        lines = cv2.HoughLines(image, 1, np.pi / 180, 85)

        return image, lines

    @staticmethod
    def __plot_canny_hough_results(image, edges, lines):
        """
        Plot the results of the Canny edge detection and Hough line detection.
        :param image: The original image used for Hough transformation
        :param edges: The edges detected by the Canny edge detector
        :param lines: The lines detected by the Hough transformation
        """
        # Plot the original image and Canny edges
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(edges, cmap='gray')
        plt.title('Canny Edge Detection')
        plt.axis('off')

        # Create a copy of the original image to draw the Hough lines
        hough_image = np.copy(image)
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(hough_image, (x1, y1), (x2, y2), 255, 1)

        plt.subplot(1, 2, 2)
        plt.imshow(hough_image, cmap='gray')
        plt.title('Hough Line Detection')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

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
        min_x = int(np.min(scanned_points[:, 0] * HoughTransformation.__scale_factor))
        min_y = int(np.min(scanned_points[:, 1] * HoughTransformation.__scale_factor))
        offset_x = -min_x if min_x < 0 else 0
        offset_y = -min_y if min_y < 0 else 0
        offset_x += HoughTransformation.__padding
        offset_y += HoughTransformation.__padding

        # Calculate the original points
        for x, y in cluster_centers:
            original_x = (x - offset_x) / HoughTransformation.__scale_factor
            original_y = (y - offset_y) / HoughTransformation.__scale_factor
            original_points.append((original_x, original_y))

        return original_points

log = np.array([
    [ 5.46375518e-17, -8.92298937e-01,
[ 1.57544965e-02, -9.02574500e-01],
[ 3.18861136e-02, -9.13098826e-01],
[ 4.84189020e-02, -9.23887686e-01],
[ 6.53786285e-02, -9.34957946e-01],
[ 8.27929519e-02, -9.46327770e-01],
[ 1.00691601e-01, -9.58016589e-01],
[ 1.19106605e-01, -9.70045455e-01],
[ 1.38072514e-01, -9.82436983e-01],
[ 1.57626660e-01, -9.95215563e-01],
[ 1.77809532e-01, -1.00840797e+00],
[ 1.98664989e-01, -1.02204276e+00],
[ 2.20240744e-01, -1.03615123e+00],
[ 2.42588803e-01, -1.05076755e+00],
[ 2.65765942e-01, -1.06592897e+00],
[ 2.89834260e-01, -1.08167618e+00],
[ 3.14861836e-01, -1.09805371e+00],
[ 3.40923639e-01, -1.11511098e+00],
[ 3.68102154e-01, -1.13290194e+00],
[ 3.96488549e-01, -1.15148636e+00],
[ 4.26183863e-01, -1.17093054e+00],
[ 4.57300325e-01, -1.19130808e+00],
[ 4.89963084e-01, -1.21270119e+00],
[ 5.24311850e-01, -1.23520131e+00],
[ 5.60503564e-01, -1.25891162e+00],
[ 5.98714578e-01, -1.28394756e+00],
[ 6.39144376e-01, -1.31044017e+00],
[ 6.82018952e-01, -1.33853756e+00],
[ 7.27595889e-01, -1.36840884e+00],
[ 7.76169602e-01, -1.40024703e+00],
[ 8.28078628e-01, -1.43427426e+00],
[ 8.83713957e-01, -1.47074701e+00],
[ 9.43529744e-01, -1.50996323e+00],
[ 1.00805645e+00, -1.55227081e+00],
[ 1.07791799e+00, -1.59807914e+00],
[ 1.15385286e+00, -1.64787267e+00],
[ 1.23674185e+00, -1.70222912e+00],
[ 1.32764446e+00, -1.76184370e+00],
[ 1.40030625e+00, -1.79231027e+00],
[ 1.42308000e+00, -1.75735745e+00],
[ 1.44560256e+00, -1.72280205e+00],
[ 1.46789185e+00, -1.68861641e+00],
[ 1.48996545e+00, -1.65477428e+00],
[ 1.51183964e+00, -1.62124952e+00],
[ 1.53353088e+00, -1.58801771e+00],
[ 1.55505467e+00, -1.55505467e+00],
[ 1.57642593e+00, -1.52233683e+00],
[ 1.59765993e+00, -1.48984199e+00],
[ 1.61877051e+00, -1.45754751e+00],
[ 1.63977221e+00, -1.42543223e+00],
[ 1.66067859e+00, -1.39347479e+00],
[ 1.68150320e+00, -1.36165444e+00],
[ 1.70225942e+00, -1.32995082e+00],
[ 1.72296036e+00, -1.29834376e+00],
[ 1.74361908e+00, -1.26681342e+00],
[ 1.76424828e+00, -1.23533994e+00],
[ 1.78486106e+00, -1.20390399e+00],
[ 1.80547007e+00, -1.17248597e+00],
[ 1.82608805e+00, -1.14106646e+00],
[ 1.84672772e+00, -1.10962596e+00],
[ 1.86740196e+00, -1.07814503e+00],
[ 1.88812371e+00, -1.04660406e+00],
[ 1.90890591e+00, -1.01498328e+00],
[ 1.92976156e+00, -9.83262625e-01],
[ 1.95070415e+00, -9.51421987e-01],
[ 1.97174728e+00, -9.19440856e-01],
[ 1.99290460e+00, -8.87298297e-01],
[ 2.01419012e+00, -8.54972981e-01],
[ 2.03561835e+00, -8.22443200e-01],
[ 2.05720415e+00, -7.89686687e-01],
[ 2.07896225e+00, -7.56680377e-01],
[ 2.10090880e+00, -7.23400913e-01],
[ 2.12305937e+00, -6.89823807e-01],
[ 2.14543095e+00, -6.55924068e-01],
[ 2.10176989e+00, -6.02672818e-01],
[ 1.99970844e+00, -5.35820260e-01],
[ 1.90792847e+00, -4.75699995e-01],
[ 1.82490305e+00, -4.21312065e-01],
[ 1.74939292e+00, -3.71844943e-01],
[ 1.68038192e+00, -3.26633158e-01],
[ 1.61498245e+00, -2.84764979e-01],
[ 1.55679190e+00, -2.46571613e-01],
[ 1.50294161e+00, -2.11224668e-01],
[ 1.45293421e+00, -1.78397889e-01],
[ 1.40634540e+00, -1.47812858e-01],
[ 1.36281077e+00, -1.19230493e-01],
[ 1.32201548e+00, -9.24443282e-02],
[ 1.28368641e+00, -6.72751542e-02],
[ 1.24758527e+00, -4.35666378e-02],
[ 1.21350317e+00, -2.11817765e-02],
[ 1.18125653e+00,  0.00000000e+00],
[ 1.15068264e+00,  2.00852403e-02],
[ 1.12163780e+00,  3.91684550e-02],
[ 1.09399341e+00,  5.73337652e-02],
[ 1.06763523e+00,  7.46563279e-02],
[ 1.04246048e+00,  9.12034741e-02],
[ 1.01837695e+00,  1.07035731e-01],
[ 9.95301486e-01,  1.22207656e-01],
[ 9.73158910e-01,  1.36768565e-01],
[ 9.51881111e-01,  1.50763157e-01],
[ 9.31406232e-01,  1.64232049e-01],
[ 9.11677859e-01,  1.77212224e-01],
[ 8.92644539e-01,  1.89737454e-01],
[ 8.74259171e-01,  2.01838633e-01],
[ 8.56478617e-01,  2.13544103e-01],
[ 8.39263243e-01,  2.24879908e-01],
[ 8.22576458e-01,  2.35870004e-01],
[ 8.06384583e-01,  2.46536508e-01],
[ 7.90656554e-01,  2.56899887e-01],
[ 7.75363435e-01,  2.66979041e-01],
[ 7.60478512e-01,  2.76791542e-01],
[ 7.45976914e-01,  2.86353708e-01],
[ 7.31835524e-01,  2.95680745e-01],
[ 7.18032769e-01,  3.04786828e-01],
[ 7.04548529e-01,  3.13685215e-01],
[ 6.91363864e-01,  3.22388264e-01],
[ 6.78461088e-01,  3.30907583e-01],
[ 6.65823709e-01,  3.39254125e-01],
[ 6.53436013e-01,  3.47438091e-01],
[ 6.41283380e-01,  3.55469182e-01],
[ 6.29351972e-01,  3.63356531e-01],
[ 6.17628578e-01,  3.71108690e-01],
[ 6.06100870e-01,  3.78733858e-01],
[ 5.94757094e-01,  3.86239773e-01],
[ 5.83586066e-01,  3.93633772e-01],
[ 5.72577222e-01,  4.00922887e-01],
[ 5.61720363e-01,  4.08113733e-01],
[ 5.51005852e-01,  4.15212691e-01],
[ 5.40424457e-01,  4.22225860e-01],
[ 5.29967302e-01,  4.29159060e-01],
[ 5.19625953e-01,  4.36017946e-01],
[ 5.09392179e-01,  4.42807865e-01],
[ 4.99258084e-01,  4.49533998e-01],
[ 4.89216145e-01,  4.56201435e-01],
[ 4.79258977e-01,  4.62815015e-01],
[ 4.69379421e-01,  4.69379421e-01],
[ 4.59570612e-01,  4.75899300e-01],
[ 4.49825725e-01,  4.82379033e-01],
[ 4.40138252e-01,  4.88823051e-01],
[ 4.30501706e-01,  4.95235562e-01],
[ 4.20909854e-01,  5.01620830e-01],
[ 4.11356427e-01,  5.07982883e-01],
[ 4.01835350e-01,  5.14325794e-01],
[ 3.92340643e-01,  5.20653619e-01],
[ 3.82866304e-01,  5.26970259e-01],
[ 3.73406444e-01,  5.33279669e-01],
[ 3.63955166e-01,  5.39585724e-01],
[ 3.54506688e-01,  5.45892428e-01],
[ 3.45055091e-01,  5.52203576e-01],
[ 3.35594575e-01,  5.58523165e-01],
[ 3.26119214e-01,  5.64855048e-01],
[ 3.16623130e-01,  5.71203247e-01],
[ 3.07100350e-01,  5.77571756e-01],
[ 2.97544795e-01,  5.83964541e-01],
[ 2.87950402e-01,  5.90385816e-01],
[ 2.78310883e-01,  5.96839615e-01],
[ 2.68619972e-01,  6.03330335e-01],
[ 2.58871190e-01,  6.09862305e-01],
[ 2.49057862e-01,  6.16439840e-01],
[ 2.39173275e-01,  6.23067684e-01],
[ 2.29210399e-01,  6.29750396e-01],
[ 2.19162065e-01,  6.36492852e-01],
[ 2.09020883e-01,  6.43300132e-01],
[ 1.98779145e-01,  6.50177287e-01],
[ 1.88428919e-01,  6.57129733e-01],
[ 1.77961959e-01,  6.64163072e-01],
[ 1.67369656e-01,  6.71283027e-01],
[ 1.56643080e-01,  6.78495721e-01],
[ 1.45772883e-01,  6.85807493e-01],
[ 1.34749256e-01,  6.93224828e-01],
[ 1.23561969e-01,  7.00754746e-01],
[ 1.12200259e-01,  7.08404558e-01],
[ 1.00652794e-01,  7.16181841e-01],
[ 8.89076516e-02,  7.24094715e-01],
[ 7.69522508e-02,  7.32151760e-01],
[ 6.47732704e-02,  7.40361869e-01],
[ 5.23566300e-02,  7.48734692e-01],
[ 3.96873760e-02,  7.57280247e-01],
[ 2.67496383e-02,  7.66009416e-01],
[ 1.35265154e-02,  7.74933550e-01]
])

# Beispielaufruf
if __name__ == "__main__":
    corners = HoughTransformation.get_landmarks(log)

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