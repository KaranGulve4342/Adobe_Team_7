import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import EllipseModel


def normalize_rounded_rectangle(coords):
    min_bounds = np.min(coords, axis=0)
    max_bounds = np.max(coords, axis=0)
    rect_width = max_bounds[0] - min_bounds[0]
    rect_height = max_bounds[1] - min_bounds[1]

    # Calculate the corner radius
    radius = min(rect_width, rect_height) * 0.1

    # Define the corner points
    corner_points = [
        [min_bounds[0] + radius, min_bounds[1] + radius],
        [max_bounds[0] - radius, min_bounds[1] + radius],
        [max_bounds[0] - radius, max_bounds[1] - radius],
        [min_bounds[0] + radius, max_bounds[1] - radius]
    ]

    # Create the rectangle with rounded corners
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

import numpy as np

def normalize_line(coords):
    if len(coords) < 2:
        return np.array([])  # Not enough points to form a line

    # Fit a line to the coordinates
    start_point, end_point = coords[0], coords[-1]
    x_values = np.array([start_point[0], end_point[0]])
    y_values = np.array([start_point[1], end_point[1]])

    return np.column_stack((x_values, y_values))


import numpy as np
from skimage.measure import EllipseModel

def adjust_ellipse(coords):
    if len(coords) < 5:
        return np.array([])  # Not enough points to fit an ellipse

    # Fit an ellipse to the coordinates
    ellipse_model = EllipseModel()
    ellipse_model.estimate(coords)
    center_x, center_y, semi_major, semi_minor, angle = ellipse_model.params

    num_points = len(coords)
    angle_values = np.linspace(0, 2 * np.pi, num_points)
    x_values = center_x + semi_major * np.cos(angle_values) * np.cos(angle) - semi_minor * np.sin(angle_values) * np.sin(angle)
    y_values = center_y + semi_major * np.cos(angle_values) * np.sin(angle) + semi_minor * np.sin(angle_values) * np.cos(angle)

    return np.column_stack((x_values, y_values))
