import cv2
import numpy as np
import matplotlib.pyplot as plt


class Landmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def hough_line_detection(image):
    # Verwende die Hough-Transformation, um Linien zu erkennen
    edges = cv2.Canny(image, 80, 200, apertureSize=3)  # Niedrigere Schwellenwerte für empfindlichere Kanten
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 30)  # Niedrigerer threshold für mehr Linien
    return lines, edges


def find_intersections(lines):
    # Finde die Schnittpunkte zwischen allen Linien
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            rho1, theta1 = lines[i][0]
            rho2, theta2 = lines[j][0]

            # Berechne die Schnittpunkte
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])

            # Lösen des linearen Gleichungssystems
            if np.linalg.det(A) != 0:  # Falls die Matrizen invertierbar sind
                intersection = np.linalg.solve(A, b)
                intersections.append(intersection.flatten())

    return np.array(intersections)


def get_landmarks(scanned_points: np.ndarray, plot_result=True):
    # Schritt 1: Finde den minimalen und maximalen Wert der Punkte, um das Bild korrekt zu erstellen
    min_x = int(np.min(scanned_points[:, 0]))
    min_y = int(np.min(scanned_points[:, 1]))
    max_x = int(np.max(scanned_points[:, 0]))
    max_y = int(np.max(scanned_points[:, 1]))

    # Verschiebung berechnen, um alle Punkte ins positive Koordinatensystem zu bringen
    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0

    # Neues Bild erstellen, welches alle Punkte enthält
    width = max_x + offset_x + 10
    height = max_y + offset_y + 10
    image = np.zeros((height, width), dtype=np.uint8)

    # Schritt 2: Punkte in das Bild zeichnen (unter Berücksichtigung der Verschiebung)
    for point in scanned_points:
        x = int(point[0]) + offset_x
        y = int(point[1]) + offset_y
        cv2.circle(image, (x, y), 2, 255, -1)

    # Schritt 3: Hough-Transformation zur Linienerkennung
    lines, edges = hough_line_detection(image)

    # Schritt 4: Finde die Schnittpunkte (Eckenerkennung)
    landmarks = []
    if lines is not None:
        intersections = find_intersections(lines)

        # Wandeln Sie Schnittpunkte in Landmark-Objekte um (unter Berücksichtigung der Verschiebung)
        for point in intersections:
            x = point[0] - offset_x
            y = point[1] - offset_y
            landmarks.append(Landmark(x=x, y=y))

    # Schritt 5: Optionales Plotten des Bildes mit erkannten Linien und Punkten
    if plot_result:
        plt.figure(figsize=(10, 10))

        # Zeige die ursprünglichen Punkte an
        plt.subplot(1, 2, 1)
        plt.title("Original Points and Edges")
        plt.imshow(edges, cmap='gray')

        # Zeichne die Linien auf das Bild
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
                cv2.line(image, (x1, y1), (x2, y2), 255, 1)

        # Zeige das Bild mit den Linien an
        plt.subplot(1, 2, 2)
        plt.title("Detected Lines and Points")
        plt.imshow(image, cmap='gray')
        plt.show()

    return landmarks


# Beispielaufruf
if __name__ == "__main__":
    # Beispiel-Punktwolke (sehr dichte Punkte)
    scanned_points = np.array([
                                  (x, 50) for x in range(0, 71)
                              ] + [
                                  (70, y) for y in range(50, -1, -1)
                              ])

    landmarks = get_landmarks(scanned_points)

    # Ausgabe der gefundenen Landmarken
    print('Landmarken:')
    for landmark in landmarks:
        print(f"Landmark: x={landmark.x:.2f}, y={landmark.y:.2f}")
