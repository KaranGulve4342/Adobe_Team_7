from tensorflow.keras.models import load_model
import svgwrite
import cairosvg
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from src.model_creation_dc import load_images_from_directory

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def polylines2svg(paths_XYs, svg_path):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)
    
    # Create a new SVG drawing
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()
    
    colours = ['red', 'green', 'blue', 'yellow', 'black']  # Define some colors for paths
    
    for i, path in enumerate(paths_XYs):
        path_data = []
        c = colours[i % len(colours)]
        for XY in path:
            path_data.append(("M", (XY[0, 0], XY[0, 1])))
            for j in range(1, len(XY)):
                path_data.append(("L", (XY[j, 0], XY[j, 1])))
            if not np.allclose(XY[0], XY[-1]):
                path_data.append(("Z", None))
        group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))
    
    dwg.add(group)
    dwg.save()
    
    png_path = svg_path.replace('.svg', '.png')
    fact = max(1, 1024 // min(H, W))
    cairosvg.svg2png(url=svg_path, write_to=png_path, parent_width=W, parent_height=H,
                     output_width=fact * W, output_height=fact * H, background_color='white')
    return

# Load the trained model
model = load_model('./models/shapes_created_model.h5')

# Load and process the CSV file
csv_path = 'frag0.csv'
paths_XYs = read_csv(csv_path)

# Convert the processed data into the format expected by the model
# Assuming the model expects 128x128 grayscale images
# You may need to adjust this part based on your specific preprocessing requirements
new_data = np.zeros((len(paths_XYs), 128, 128, 1))
for i, path_XYs in enumerate(paths_XYs):
    for path in path_XYs:
        for XY in path:
            for x, y in XY:
                new_data[i, int(y), int(x), 0] = 1

# Normalize the data
new_data = new_data / 255.0

# Make predictions
predictions = model.predict(new_data)

# Get the class with the highest probability
predicted_classes = np.argmax(predictions, axis=1)

print(predicted_classes)

# Optionally, visualize the shapes
svg_path = 'output.svg'
polylines2svg(paths_XYs, svg_path)