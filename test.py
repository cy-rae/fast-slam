from bokeh.plotting import figure, output_file, save

# Erstelle eine neue Plot-Figur
p = figure(
    title="Map created by the FastSLAM 2.0 algorithm",
    x_range=(-6, 6), y_range=(-6, 6),
    x_axis_label='X-axis', y_axis_label='Y-axis'
)

# Add a point at (0,0) using scatter
p.scatter([0.0], [0.0], size=10, color="black")

# Speichere die Plot-Figur als HTML-Datei
output_file("map.html")
save(p)
