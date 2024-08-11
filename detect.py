import sys
import cv2
import numpy as np

sift_detector = cv2.SIFT_create()

def are_points_close(point1, point2, tolerance=4.0):
    """Checks if the points point1, point2 are within tolerance distance of each other."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) < tolerance

def scale_factor(size1, size2, sigma=1):
    """Computes the scale factor 'S' based on the research paper."""
    ratio = (-abs(size1 - size2)) / (sigma * (size1 + size2))
    return np.exp(ratio ** 2)

def reisfeld_measure(angle1, angle2, theta):
    """Computes the Reisfeld measure for symmetry."""
    return 1 - np.cos(angle1 + angle2 - 2 * theta)

def compute_midpoint(point1, point2):
    """Calculates the midpoint between two points."""
    return ( (point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2 )

def angle_with_horizontal(point1, point2):
    """Calculates the angle of the line connecting point1 and point2 with the x-axis."""
    delta_x, delta_y = point1[0] - point2[0], point1[1] - point2[1]
    if delta_x == 0:
        return np.pi / 2
    angle = np.arctan(delta_y / delta_x)
    if angle < 0:
        angle += np.pi
    return angle

def detect_symmetry(image):
    """Performs symmetry detection on the image."""
    mirrored_image = np.fliplr(image)
    keypoints1, descriptors1 = sift_detector.detectAndCompute(image, None)
    keypoints2, descriptors2 = sift_detector.detectAndCompute(mirrored_image, None)

    for kp1, kp2 in zip(keypoints1, keypoints2):
        kp1.angle = np.deg2rad(kp1.angle)
        kp2.angle = np.deg2rad(kp2.angle)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    distances = np.zeros(len(matches))
    angles = np.zeros(len(matches))
    weights = np.zeros(len(matches))
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

        if are_points_close(kp1.pt, kp2.pt):
            kp2 = kp2_alt
            good_matches.append(second_match)
        else:
            good_matches.append(match)

        theta = angle_with_horizontal(kp1.pt, kp2.pt)
        x_center, y_center = compute_midpoint(kp1.pt, kp2.pt)
        radius = x_center * np.cos(theta) + y_center * np.sin(theta)
        measure = reisfeld_measure(kp1.angle, kp2.angle, theta) * scale_factor(
            kp1.size, kp2.size
        )

        distances[i] = radius
        angles[i] = theta
        weights[i] = measure
        i += 1

    good_matches = sorted(good_matches, key=lambda x: x.distance)

    def draw_line(radius, angle):
        """Draws the detected line on the image."""
        if np.pi / 4 < angle < 3 * (np.pi / 4):
            for x in range(len(image[0])):
                y = int((radius - x * np.cos(angle)) / np.sin(angle))
                if 0 <= y < len(image):
                    image[y][x] = 100
        else:
            for y in range(len(image)):
                x = int((radius - y * np.sin(angle)) / np.cos(angle))
                if 0 <= x < len(image[0]):
                    image[y][x] = 100

    matched_image = cv2.drawMatches(image, keypoints1, mirrored_image, keypoints2, good_matches[:15], None, flags=2)
    draw_line(radius, angle)
    cv2.imshow('Symmetry Detection', image)
    cv2.waitKey(0)

def main():
    """Main function to handle command-line arguments and execute appropriate functionality."""
    arg_count = len(sys.argv)
    if not (arg_count == 2 or arg_count == 4 or arg_count == 5):
        print("Usage: python3 detect.py IMAGE [r] [theta]")
        return
    if arg_count == 2:
        detect_symmetry(cv2.imread(sys.argv[1], 0))
    elif arg_count == 4:
        img = cv2.imread(sys.argv[1], 0)
        draw_line(img, float(sys.argv[2]), float(sys.argv[3]))
        cv2.imshow("Image with Line", img)
        cv2.waitKey(0)
    else:
        img = cv2.imread(sys.argv[1], 0)
        draw_line(img, float(sys.argv[2]), float(sys.argv[3]))
        cv2.imwrite("{}".format(sys.argv[4]), img)

if __name__ == "__main__":
    main()
