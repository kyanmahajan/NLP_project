import sys
from PIL import Image
import numpy as np

def load_and_preprocess_image(image_path):
    """Load the image and convert it to a binary format."""
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        threshold = np.mean(img_array) * 0.8
        binary = (img_array < threshold).astype(np.uint8)
        return binary
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        return None

def detect_intersections(binary):
    """Detect intersections in the binary image."""
    height, width = binary.shape
    visited = np.zeros_like(binary)
    intersections = []

    for y in range(3, height - 3):
        for x in range(3, width - 3):
            if binary[y, x] == 0 or visited[y, x] == 1:
                continue

            win_3x3 = binary[y-1:y+2, x-1:x+2]
            win_5x5 = binary[y-2:y+3, x-2:x+3]
            win_7x7 = binary[y-3:y+4, x-3:x+4]

            sum_3x3 = np.sum(win_3x3) - binary[y, x]
            sum_5x5 = np.sum(win_5x5) - binary[y, x]
            sum_7x7 = np.sum(win_7x7) - binary[y, x]

            if sum_3x3 >= 3 and sum_5x5 >= 6 and sum_7x7 <= 30:
                intersections.append((x, y))
                visited[y-3:y+4, x-3:x+4] = 1

    return intersections

def cluster_points(intersections):
    """Cluster close points to avoid duplicates due to thick lines."""
    final_points = []
    dist_sq = 36  # within 6px

    for x1, y1 in intersections:
        is_new = True
        for x2, y2 in final_points:
            if (x1 - x2)**2 + (y1 - y2)**2 < dist_sq:
                is_new = False
                break
        if is_new:
            final_points.append((x1, y1))

    return final_points

def count_intersections(image_path):
    """Count the number of intersections in the image."""
    binary = load_and_preprocess_image(image_path)
    if binary is None:
        return 0

    intersections = detect_intersections(binary)
    final_points = cluster_points(intersections)
    return len(final_points)

if __name__ == "__main__":
    filename = input().strip()
    print(count_intersections(filename))
