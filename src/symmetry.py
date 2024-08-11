import numpy as np
import svgwrite
import cairosvg
from PIL import Image

# Define the colors used for the polylines
line_colors = ['#000000']  # Adjust this as needed, currently set to black for grayscale

def paths_to_svg(polyline_paths, svg_file):
    max_width, max_height = 0, 0
    for path in polyline_paths:
        for coordinates in path:
            max_width, max_height = max(max_width, np.max(coordinates[:, 0])), max(max_height, np.max(coordinates[:, 1]))
    padding = 0.1
    width, height = int(max_width + padding * max_width), int(max_height + padding * max_height)

    # Create a new SVG drawing with specified size
    drawing = svgwrite.Drawing(svg_file, profile='tiny', size=(width, height), shape_rendering='crispEdges')
    group = drawing.g()

    for index, path in enumerate(polyline_paths):
        path_commands = []
        color = line_colors[index % len(line_colors)]
        for coordinates in path:
            path_commands.append(("M", (coordinates[0, 0], coordinates[0, 1])))
            for j in range(1, len(coordinates)):
                path_commands.append(("L", (coordinates[j, 0], coordinates[j, 1])))
        group.add(drawing.path(d=path_commands, fill='none', stroke=color, stroke_width=2))

    drawing.add(group)
    drawing.save()

    png_file = svg_file.replace('.svg', '.png')
    scale_factor = max(1, 1024 // min(height, width))

    # Convert the SVG to PNG
    cairosvg.svg2png(url=svg_file, write_to=png_file, output_width=scale_factor * width, output_height=scale_factor * height, background_color='white')

    # Convert the PNG to grayscale
    image = Image.open(png_file).convert('L')
    image.save(png_file)
    return png_file

def load_csv(csv_file):
    data = np.genfromtxt(csv_file, delimiter=',')
    polyline_paths = []

    for i in np.unique(data[:, 0]):
        subset = data[data[:, 0] == i][:, 1:]
        path_segments = []
        for j in np.unique(subset[:, 0]):
            segment = subset[subset[:, 0] == j][:, 1:]
            path_segments.append(segment)
        polyline_paths.append(path_segments)

    return polyline_paths

# Example usage
csv_file = 'C:\\Users\\KARAN\\Desktop\\Adobe_Team_7\\problems\\occlusion1.csv'
polyline_paths = load_csv(csv_file)
svg_file = 'polylines.svg'
png_file = paths_to_svg(polyline_paths, svg_file)
print(f"Saved grayscale PNG to {png_file}")