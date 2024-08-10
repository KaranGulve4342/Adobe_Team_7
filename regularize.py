import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import EllipseModel
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev


def normalize_rounded_rectangle(coords):
    min_bounds = np.min(coords, axis=0)
    max_bounds = np.max(coords, axis=0)
    rect_width = max_bounds[0] - min_bounds[0]
    rect_height = max_bounds[1] - min_bounds[1]

    radius = min(rect_width, rect_height) * 0.1

    corner_points = [
        [min_bounds[0] + radius, min_bounds[1] + radius],
        [max_bounds[0] - radius, min_bounds[1] + radius],
        [max_bounds[0] - radius, max_bounds[1] - radius],
        [min_bounds[0] + radius, max_bounds[1] - radius]
    ]

    x_coords = [point[0] for point in corner_points] + [corner_points[0][0]]
    y_coords = [point[1] for point in corner_points] + [corner_points[0][1]]

    return np.column_stack((x_coords, y_coords))

def standardize_polygon(vertices):
    centroid = np.mean(vertices, axis=0)
    avg_radius = np.mean(np.linalg.norm(vertices - centroid, axis=1))
    num_vertices = len(vertices)

    angle_steps = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
    x_coords = centroid[0] + avg_radius * np.cos(angle_steps)
    y_coords = centroid[1] + avg_radius * np.sin(angle_steps)

    return np.column_stack((x_coords, y_coords))

def normalize_line(coords):
    if len(coords) < 2:
        return np.array([])  

    start_point, end_point = coords[0], coords[-1]
    x_values = np.array([start_point[0], end_point[0]])
    y_values = np.array([start_point[1], end_point[1]])

    return np.column_stack((x_values, y_values))

def adjust_ellipse(coords):
    if len(coords) < 5:
        return np.array([]) 

    ellipse_model = EllipseModel()
    ellipse_model.estimate(coords)
    center_x, center_y, semi_major, semi_minor, angle = ellipse_model.params

    num_points = len(coords)
    angle_values = np.linspace(0, 2 * np.pi, num_points)
    x_values = center_x + semi_major * np.cos(angle_values) * np.cos(angle) - semi_minor * np.sin(angle_values) * np.sin(angle)
    y_values = center_y + semi_major * np.cos(angle_values) * np.sin(angle) + semi_minor * np.sin(angle_values) * np.cos(angle)

    return np.column_stack((x_values, y_values))

def adjust_circle(coords, img_width, img_height, scale_factor=0.9):
    center_point = np.mean(coords, axis=0)
    avg_radius = np.median(np.linalg.norm(coords - center_point, axis=1))

    # Calculate the scale based on the minimum dimension
    min_dimension = min(img_width, img_height)
    scaling_factor = (min_dimension * scale_factor) / (2 * avg_radius)

    num_coords = len(coords)
    angles = np.linspace(0, 2 * np.pi, num_coords, endpoint=False)
    x_values = center_point[0] + avg_radius * scaling_factor * np.cos(angles)
    y_values = center_point[1] + avg_radius * scaling_factor * np.sin(angles)

    return np.column_stack((x_values, y_values))

def adjust_rectangle(corner_points, img_width, img_height, scale_factor=0.9):
    min_point = np.min(corner_points, axis=0)
    max_point = np.max(corner_points, axis=0)

    rect_width = max_point[0] - min_point[0]
    rect_height = max_point[1] - min_point[1]

    scale_x = (img_width * scale_factor) / rect_width
    scale_y = (img_height * scale_factor) / rect_height

    rect_center = np.mean(corner_points, axis=0)

    x_coords = [min_point[0], max_point[0], max_point[0], min_point[0], min_point[0]]
    y_coords = [min_point[1], min_point[1], max_point[1], max_point[1], min_point[1]]

    x_scaled = rect_center[0] + (np.array(x_coords) - rect_center[0]) * scale_x
    y_scaled = rect_center[1] + (np.array(y_coords) - rect_center[1]) * scale_y

    return np.column_stack((x_scaled, y_scaled))

def adjust_starshape(points):
    star_center = np.mean(points, axis=0)
    max_distance = np.max(np.linalg.norm(points - star_center, axis=1))
    inner_distance = max_distance * 0.4 

    num_vertices = 10 
    angle_values = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
    radii = np.zeros(num_vertices)
    radii[::2] = max_distance  
    radii[1::2] = inner_distance  

    x_coords = star_center[0] + radii * np.cos(angle_values)
    y_coords = star_center[1] + radii * np.sin(angle_values)

    x_coords = np.append(x_coords, x_coords[0])
    y_coords = np.append(y_coords, y_coords[0])

    return np.column_stack((x_coords, y_coords))

def compute_perimeter(convex_hull):
    total_perimeter = 0
    num_vertices = len(convex_hull.vertices)
    
    for i in range(num_vertices):
        point1 = convex_hull.points[convex_hull.vertices[i]]
        point2 = convex_hull.points[convex_hull.vertices[(i + 1) % num_vertices]]
        total_perimeter += np.linalg.norm(point1 - point2)
    
    return total_perimeter

def get_vertex_count(coords):
    convex_hull = ConvexHull(coords)
    return len(convex_hull.vertices)

def detect_shape(coords):
    print(len(coords))
    if len(coords) < 3:
        return 'unknown'  # Not enough points to determine shape

    convex_hull = ConvexHull(coords)
    hull_area = convex_hull.volume
    hull_perimeter = compute_perimeter(convex_hull)

    if hull_perimeter == 0:
        return 'unknown'

    shape_compactness = 4 * np.pi * hull_area / (hull_perimeter ** 2)

    print(f"Compactness: {shape_compactness}, Hull Area: {hull_area}, Hull Perimeter: {hull_perimeter}")

    if shape_compactness > 0.88:
        print("Identified as circle")
        return 'circle'

    shape_aspect_ratio = compute_aspect_ratio(coords)
    if 0.8 < shape_aspect_ratio < 1.2:
        print("Identified as rectangle")
        return 'rectangle'

    num_vertices = get_vertex_count(coords)

    # Adjust aspect ratio threshold for star
    if num_vertices > 5 and (shape_aspect_ratio < 0.9 or shape_aspect_ratio > 2) and shape_compactness < 0.9:
        print("Identified as star")
        return 'star'

    return 'unknown'

def compute_max_dimensions(coords):
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    dimension_width = max_coords[0] - min_coords[0]
    dimension_height = max_coords[1] - min_coords[1]
    return dimension_width, dimension_height

def compute_aspect_ratio(coords):
    hull = ConvexHull(coords)
    hull_coords = hull.points[hull.vertices]

    # Calculate distances between vertices
    distances = [np.linalg.norm(hull_coords[i] - hull_coords[(i + 1) % len(hull_coords)]) for i in range(len(hull_coords))]
    dimension_width, dimension_height = sorted(distances)[:2]

    return max(dimension_width, dimension_height) / min(dimension_width, dimension_height)

def standardize_shape(coords):
    if len(coords) == 0:
        return np.array([])  # Return empty array for empty input

    # Determine the maximum extents from the irregular shape
    max_width, max_height = compute_max_dimensions(coords)

    shape_category = detect_shape(coords)

    if shape_category == 'circle':
        return adjust_circle(coords, max_width, max_height)
    elif shape_category == 'rectangle':
        return adjust_rectangle(coords, max_width, max_height)
    elif shape_category == 'star':
        return adjust_starshape(coords)
    else:
        return bezier_curve_fit(coords)
    
def bezier_curve_fit(points, num_points=100):
    tck, u = splprep(points.T, s=0)
    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.vstack((x_new, y_new)).T

