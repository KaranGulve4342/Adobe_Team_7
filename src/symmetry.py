import numpy as np
import svgwrite
import cairosvg
from PIL import Image
import cv2
import matplotlib.pyplot as plt

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


# Define the required functions
def points_close(pt1, pt2, threshold=5.0):
    """Returns True if the two points are within a certain threshold distance."""
    return np.linalg.norm(np.array(pt1) - np.array(pt2)) < threshold

def angle_with_horizontal(pt1, pt2):
    """Calculates the angle of the line connecting pt1 and pt2 with the x-axis."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    return np.arctan2(dy, dx)

def midpoint(pt1, pt2):
    """Calculates the midpoint between two points."""
    return ( (pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2 )

def reisfeld_measure(angle1, angle2, theta):
    """Computes the Reisfeld measure for symmetry."""
    return 1 - np.cos(2 * (angle1 - angle2 - theta))

def scale_factor(size1, size2):
    """Computes the scale factor for symmetry detection."""
    return min(size1, size2) / max(size1, size2)

def detect_symmetry(image):
    """Performs the symmetry detection on image and plots the hexbin plot."""
    mirrored_image = np.fliplr(image)
    keypoints1, descriptors1 = sift.detectAndCompute(image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(mirrored_image, None)
    for kp, mkp in zip(keypoints1, keypoints2):
        kp.angle = np.deg2rad(kp.angle)
        mkp.angle = np.deg2rad(mkp.angle)
    bf_matcher = cv2.BFMatcher()
    matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)
    distances = []
    angles = []
    match_weights = []
    good_matches = []

    for match, second_match in matches:
        kp1 = keypoints1[match.queryIdx]
        kp2 = keypoints2[match.trainIdx]
        kp2_alt = keypoints2[second_match.trainIdx]
        kp2_alt.angle = np.pi - kp2_alt.angle
        kp2.angle = np.pi - kp2.angle
        if kp2.angle < 0.0:
            kp2.angle += 2 * np.pi
        if kp2_alt.angle < 0.0:
            kp2_alt.angle += 2 * np.pi
        kp2.pt = (mirrored_image.shape[1] - kp2.pt[0], kp2.pt[1])
        if points_close(kp1.pt, kp2.pt):
            kp2 = kp2_alt
            good_matches.append(second_match)
        else:
            good_matches.append(match)
        angle = angle_with_horizontal(kp1.pt, kp2.pt)
        x_center, y_center = midpoint(kp1.pt, kp2.pt)
        distance = x_center * np.cos(angle) + y_center * np.sin(angle)
        measure = reisfeld_measure(kp1.angle, kp2.angle, angle) * scale_factor(
            kp1.size, kp2.size
        )
        distances.append(distance)
        angles.append(angle)
        match_weights.append(measure)

    distances = np.array(distances)
    angles = np.array(angles)
    match_weights = np.array(match_weights)

    # Plotting the hexbin plot with interactivity
    def plot_hexbin():
        fig, ax = plt.subplots(figsize=(10, 7))
        hexbin = ax.hexbin(distances, angles, gridsize=50, cmap='Reds', mincnt=1)
        plt.title('Hexbin Plot of Symmetry Detection')
        plt.xlabel('Distance')
        plt.ylabel('Angle')
        colorbar = plt.colorbar(hexbin, ax=ax, label='Number of Votes')

        # Find the hexbins with the highest density (darkest color)
        max_counts_idx = np.argmax(hexbin.get_array())
        distance_val = hexbin.get_offsets()[max_counts_idx][0]
        angle_val = hexbin.get_offsets()[max_counts_idx][1]

        print(f"Maximum density at distance: {distance_val:.2f}, angle: {angle_val:.2f} radians")

        plt.show()

    plot_hexbin()

# Example usage
if __name__ == "__main__":
    img = cv2.imread("C:\\Users\\KARAN\\Desktop\\polylines.png", 0)  # Replace with your image path
    sift = cv2.SIFT_create()  # Initialize SIFT detector
    detect_symmetry(img)