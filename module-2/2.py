# We will implement a full pipeline:
# 1. Use the Canny Edge Detector to find clean edge maps.
# 2. Use the Probabilistic Hough Transform to detect lines in the edge map.
# 3. Use the Hough Circle Transform to detect circles.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Helper Functions ---

def display_images(images, titles, main_title):
    """Helper to display multiple images with titles."""
    num_images = len(images)
    plt.figure(figsize=(5 * num_images, 5))
    plt.suptitle(main_title, fontsize=16)
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        if len(images[i].shape) == 2:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def create_line_image():
    """Creates a sample image with several straight lines."""
    img = np.full((300, 300), 255, dtype=np.uint8)
    # Draw some lines of varying angles and positions
    cv2.line(img, (50, 50), (250, 50), 0, 3)
    cv2.line(img, (50, 100), (250, 120), 0, 3)
    cv2.line(img, (50, 50), (50, 250), 0, 3)
    cv2.line(img, (150, 50), (50, 250), 0, 3)
    cv2.line(img, (200, 200), (280, 20), 0, 3)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def create_circle_image():
    """Creates a sample image with several circles."""
    img = np.full((300, 300), 255, dtype=np.uint8)
    # Draw some circles of varying sizes and positions
    cv2.circle(img, (70, 70), 40, 0, 3)
    cv2.circle(img, (180, 120), 25, 0, 3)
    cv2.circle(img, (150, 220), 50, 0, 3)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# --- Main Lab Tasks ---

def task_1_line_detection():
    """
    Task 1: Detect lines in an image using the Canny->Hough pipeline.
    """
    print("Running Task 1: Line Detection...")
    img = create_line_image()
    img_for_drawing = img.copy()

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply Gaussian Blur to reduce noise before Canny
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Apply Canny Edge Detector
    # Tuning these thresholds is key!
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # 4. Apply Probabilistic Hough Line Transform
    # threshold: Minimum number of votes to be a line
    # minLineLength: Minimum length of a line
    # maxLineGap: Maximum gap between segments to be a single line
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80, 
        minLineLength=30,
        maxLineGap=10
    )

    # 5. Draw the detected lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_for_drawing, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(f"Found {len(lines)} lines.")
    else:
        print("Found no lines.")

    # 6. Display results
    display_images(
        [img, edges, img_for_drawing],
        ["Original Image", "Canny Edges", "Detected Lines"],
        "Task 1: Canny Edge Detector and Hough Line Transform"
    )

def task_2_circle_detection():
    """
    Task 2: Detect circles in an image using the Hough Circle Transform.
    """
    print("Running Task 2: Circle Detection...")
    img = create_circle_image()
    img_for_drawing = img.copy()

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Apply Median Blur to reduce noise before HoughCircles
    # Median blur is often effective for salt-and-pepper noise.
    gray_blurred = cv2.medianBlur(gray, 5)

    # 3. Apply Hough Circle Transform
    # dp: Inverse ratio of accumulator resolution to image resolution.
    # minDist: Minimum distance between centers of detected circles.
    # param1: Higher threshold for the internal Canny edge detector.
    # param2: Accumulator threshold for circle centers (lower means more circles).
    # minRadius, maxRadius: The range of circle sizes to look for.
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, 
                               minDist=50, param1=100, param2=30, 
                               minRadius=10, maxRadius=60)

    # 4. Draw the detected circles
    if circles is not None:
        # Convert coordinates and radius to integers
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # Draw the outer circle
            cv2.circle(img_for_drawing, center, radius, (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(img_for_drawing, center, 2, (0, 0, 255), 3)
        print(f"Found {len(circles[0])} circles.")
    else:
        print("Found no circles.")

    # 5. Display results
    display_images(
        [img, gray_blurred, img_for_drawing],
        ["Original Image", "Blurred Grayscale", "Detected Circles"],
        "Task 2: Hough Circle Transform"
    )

if __name__ == '__main__':
    task_1_line_detection()
    task_2_circle_detection()
