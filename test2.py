from PIL import Image, ImageDraw, ImageFont

# Bildgröße und Koordinatenbereich definieren
width, height = 600, 600
x_min, x_max = -6, 6
y_min, y_max = -6, 6

# Bild erstellen (weißes Hintergrund)
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)

# Achsen erstellen
# Mitte des Bildes als (0, 0) definieren
center_x = width // 2
center_y = height // 2

# Achsen zeichnen
draw.line((0, center_y, width, center_y), fill="black", width=2)  # X-Achse
draw.line((center_x, 0, center_x, height), fill="black", width=2)  # Y-Achse

# Beschriftungen für die Achsen hinzufügen
try:
    # Versuchen, eine Standardschriftart zu verwenden
    font = ImageFont.load_default()
except IOError:
    font = None

draw.text((width - 100, center_y + 10), "X-axis", fill="black", font=font)
draw.text((center_x + 10, 10), "Y-axis", fill="black", font=font)

# Titel hinzufügen
draw.text((width // 4, 10), "Map created by the FastSLAM 2.0 algorithm", fill="black", font=font)

# Punkt in der Mitte zeichnen (0,0) - 'ko'
point_radius = 5
draw.ellipse(
    (center_x - point_radius, center_y - point_radius, center_x + point_radius, center_y + point_radius),
    fill="black"
)

# Bild speichern
image.save("map.jpg", "JPEG")
