import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid
from rdp import rdp
import plot as p


def simplify_path(points_list, tolerance=1.0):
    reduced_paths = []
    for points_group in points_list:
        reduced_group = []
        for segment in points_group:
            if len(segment) > 2:
                reduced_segment = rdp(segment, epsilon=tolerance)
                reduced_group.append(reduced_segment)
            else:
                reduced_group.append(segment)
        reduced_paths.append(reduced_group)
    return reduced_paths

def reduce_path_simplicity(coordinate_paths, tolerance=1.0):
    simplified_paths = []
    for path in coordinate_paths:
        simplified_path = []
        for segment in path:
            if len(segment) > 2:
                reduced_segment = rdp(segment, epsilon=tolerance)
                simplified_path.append(reduced_segment)
            else:
                simplified_path.append(segment)
        simplified_paths.append(simplified_path)
    return simplified_paths

def fill_gaps(coordinate_paths):
    completed_paths = []
    for path in coordinate_paths:
        completed_path = []
        for segment in path:
            if len(segment) > 2:  # Only process segments with more than 2 points
                convex_hull = ConvexHull(segment)
                completed_segment = segment[convex_hull.vertices]
            else:
                completed_segment = segment
            completed_path.append(completed_segment)
        completed_paths.append(completed_path)
    return completed_paths


def generate_smooth_curve(point_set, num_samples=100):
    if len(point_set) < 2:
        return point_set
    spline_params, parameter_values = splprep([point_set[:, 0], point_set[:, 1]], s=0)
    parameter_fine = np.linspace(0, 1, num_samples)
    x_smooth, y_smooth = splev(parameter_fine, spline_params)
    return np.vstack((x_smooth, y_smooth)).T

def visualize_curves(coordinate_sets, color_palette):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for idx, coordinates in enumerate(coordinate_sets):
        color = color_palette[idx % len(color_palette)]
        for segment in coordinates:
            if len(segment) > 2:
                smooth_curve = generate_smooth_curve(segment)
                ax.plot(smooth_curve[:, 0], smooth_curve[:, 1], c=color, linewidth=2)
                ax.fill(smooth_curve[:, 0], smooth_curve[:, 1], c=color, alpha=0.3)
            else:
                ax.plot(segment[:, 0], segment[:, 1], c=color, linewidth=2)
                ax.fill(segment[:, 0], segment[:, 1], c=color, alpha=0.3)
    ax.set_aspect('equal')
    plt.show()

def evaluate_connectivity(shape_collections):
    valid_polygons = []
    for collection in shape_collections:
        for shape in collection:
            if len(shape) > 2:
                polygon = Polygon(shape)
                if polygon.is_valid:
                    valid_polygons.append(polygon)
                else:
                    fixed_polygon = make_valid(polygon)
                    valid_polygons.append(fixed_polygon)

    if not valid_polygons:
        return "Disconnected"

    try:
        multi_polygon = MultiPolygon(valid_polygons)
        combined_polygon = unary_union(multi_polygon)

        if isinstance(combined_polygon, Polygon):
            return "Connected"
        elif isinstance(combined_polygon, MultiPolygon):
            if len(combined_polygon.geoms) == 1:
                return "Connected"
            else:
                return "Disconnected"
        elif isinstance(combined_polygon, GeometryCollection):
            valid_polygons = [geom for geom in combined_polygon.geoms if isinstance(geom, Polygon) and geom.is_valid]
            if valid_polygons:
                multi_polygon = MultiPolygon(valid_polygons)
                if len(multi_polygon.geoms) == 1:
                    return "Connected"
                else:
                    return "Disconnected"
            else:
                return "Disconnected"
    except Exception as e:
        print(f"Error in connectivity analysis: {e}")
        return "Disconnected"

def handle_csv_and_fill_gaps(input_csv, color_palette):
    # Read input CSV
    path_data = p.read_csv(input_csv)

    # Apply RDP simplification
    simplified_paths = reduce_path_simplicity(path_data, tolerance=1.0)
    print("Plotting input curves...")
    visualize_curves(simplified_paths, color_palette)
    initial_connectivity = evaluate_connectivity(simplified_paths)

    # Fill gaps
    completed_paths = fill_gaps(simplified_paths)
    print("Plotting completed curves...")
    visualize_curves(completed_paths, color_palette)
    final_connectivity = evaluate_connectivity(completed_paths)

    print(f"Initial connectivity: {initial_connectivity}")
    print(f"Final connectivity: {final_connectivity}")

    if initial_connectivity == "Disconnected" and final_connectivity == "Connected":
        print("Gaps have been filled")
    else:
        print("No change in connectivity")

# Example usage
input_csv = '..\problems\occlusion1.csv'
color_palette = ['red', 'green']

handle_csv_and_fill_gaps(input_csv, color_palette)

path_data = p.read_csv(input_csv)
simplified_paths = reduce_path_simplicity(path_data, tolerance=1.0)
p.polylines_to_svg(simplified_paths, 'simplified_paths.svg', color_palette)

