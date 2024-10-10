import numpy as np


def best_fit_transform(source_points, target_points):
    """
    Compute the best fitting rotation matrix and translation vector
    that aligns the source points to the target points.
    """
    # Compute centroids of both point sets
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    # Center the points
    centered_source = source_points - centroid_source
    centered_target = target_points - centroid_target

    # Compute covariance matrix
    H = centered_source.T @ centered_target

    # Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)

    # Compute the rotation matrix
    R = Vt.T @ U.T

    # Handle reflection case (det(R) should be 1, if det(R) == -1, we correct it)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute the translation vector
    t = centroid_target.T - R @ centroid_source.T

    return R, t


def generate_test_data():
    # Definierte Source Points (z.B. ein Dreieck in 2D)
    source_points = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

    # Definiere den Winkel für die Rotation (45 Grad = Pi/4)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])  # Rotationsmatrix

    # Definiere den Translationsvektor
    t = np.array([2, 3])

    # Wende die Rotation und Translation auf die Source Points an, um die Target Points zu generieren
    target_points = (R @ source_points.T).T + t

    return source_points, target_points


# Example usage
if __name__ == "__main__":
    # Generiere neue Testdaten
    source_points, target_points = generate_test_data()

    print("Source Points:")
    print(source_points)
    print("Target Points (after rotation and translation):")
    print(target_points)

    # Berechne die Rotationsmatrix und den Translationsvektor
    R, t = best_fit_transform(source_points, target_points)

    print("Calculated Rotation matrix:")
    print(R)
    print("Calculated Translation vector:")
    print(t)