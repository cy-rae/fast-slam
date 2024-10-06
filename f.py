import numpy as np
import matplotlib.pyplot as plt

def generate_points(num_points, noise_level):
    """Generiere zufällige Punktdaten mit Rauschen"""
    x = np.linspace(0, 10, num_points)
    y = 2 * x + np.random.normal(0, noise_level, num_points)  # y = 2x + Rauschen
    return np.column_stack((x, y))

def iepf(points, threshold=1.0, min_segment_length=3):
    """IEPF-Algorithmus zur Liniensegmenterkennung"""
    lines = []
    n_points = len(points)

    # Iteriere über alle Punkte
    i = 0
    while i < n_points:
        line_segment = [points[i]]

        # Suche nach dem nächsten Punkt, der zu diesem Segment gehört
        for j in range(i + 1, n_points):
            dist = np.linalg.norm(points[j] - line_segment[-1])
            if dist < threshold:
                line_segment.append(points[j])
            else:
                break  # Wenn der Abstand zu groß ist, brich die Schleife ab

        if len(line_segment) >= min_segment_length:
            lines.append(line_segment)

        i += len(line_segment)  # Überspringe die Punkte, die bereits verarbeitet wurden

    return lines

def plot_segments(segments):
    """Zeichne die erkannten Liniensegmente"""
    for segment in segments:
        segment = np.array(segment)
        plt.plot(segment[:, 0], segment[:, 1], marker='o')

    plt.title("Erkannte Liniensegmente")
    plt.xlabel("X-Achse")
    plt.ylabel("Y-Achse")
    plt.grid()
    plt.show()

# Beispielverwendung
num_points = 100
noise_level = 0.5
points = generate_points(num_points, noise_level)

# Finde Liniensegmente
segments = iepf(points, threshold=1.0, min_segment_length=3)

# Plotte die Ergebnisse
plot_segments(segments)
