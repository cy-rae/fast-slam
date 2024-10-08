import numpy as np
import matplotlib.pyplot as plt


# Initialisiere das Bild
def init_image():
    return np.random.rand(10, 10)


# Setze die Figur und die Achsen
fig, ax = plt.subplots()

# Plotte das initiale Bild
initial_image = init_image()
img = ax.imshow(initial_image, cmap='viridis')

# Endlos-Schleife, die das Bild aktualisiert
while True:
    # Erzeuge ein neues Bild
    new_image = np.random.rand(10, 10)

    # Aktualisiere das gezeigte Bild
    img.set_array(new_image)

    # Aktualisiere die Anzeige
    plt.draw()
    plt.pause(0.2)  # Warte 200ms vor dem nächsten Update
