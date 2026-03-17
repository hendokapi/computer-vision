# We will implement and visualize:
# 1. Harris Corner Detection
# 2. HOG (Histogram of Oriented Gradients) Feature Extraction
# 3. LBP (Local Binary Patterns) Feature Extraction

import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Helper Functions ---

def display_images(images, titles, main_title, cmap_list=None):
    """Helper to display multiple images with titles and optional colormaps."""
    num_images = len(images)
    plt.figure(figsize=(5 * num_images, 5))
    plt.suptitle(main_title, fontsize=16)
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        cmap = cmap_list[i] if cmap_list else 'gray'
        if len(images[i].shape) == 2:
            plt.imshow(images[i], cmap=cmap)
        else:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def create_chessboard_image():
    """Creates a simple 8x8 chessboard image."""
    img = np.zeros((256, 256), dtype=np.uint8)
    tile_size = 32
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                img[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = 255
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def create_person_image():
    """Creates a very simple stick-figure person image for HOG."""
    img = np.full((128, 64), 255, dtype=np.uint8)
    # Head
    cv2.circle(img, (32, 24), 10, 0, -1)
    # Body
    cv2.line(img, (32, 34), (32, 80), 0, 3)
    # Arms
    cv2.line(img, (15, 50), (49, 50), 0, 3)
    # Legs
    cv2.line(img, (32, 80), (15, 110), 0, 3)
    cv2.line(img, (32, 80), (49, 110), 0, 3)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def create_texture_image():
    """Creates a sample image with simple textures."""
    img = np.zeros((128, 128), dtype=np.uint8)
    # Add some random noise texture
    noise = np.zeros_like(img, dtype=np.int8)
    cv2.randn(noise, 0, 50)
    img = cv2.add(img, noise, dtype=cv2.CV_8UC1)
    # Add some regular patterns
    for i in range(0, 128, 16):
        cv2.line(img, (i, 0), (i, 128), 128, 1)
        cv2.line(img, (0, i), (128, i), 128, 1)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# --- Main Lab Tasks ---

def task_1_harris_corners():
    """Task 1: Detects and visualizes Harris Corners."""
    print("Running Task 1: Harris Corner Detection...")
    img = create_chessboard_image()
    img_for_drawing = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # 1. Apply cv2.cornerHarris()
    # blockSize: Neighborhood size for corner detection
    # ksize: Aperture parameter for the Sobel operator
    # k: Harris detector free parameter (0.04 to 0.06 is a good range)
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    # 2. Dilate the corner response map to make corners more visible
    dst = cv2.dilate(dst, None)

    # 3. Threshold the result to get strong corners
    # We mark corners in red on the original image
    threshold = 0.01 * dst.max()
    img_for_drawing[dst > threshold] = [0, 0, 255]

    # 4. Display results
    display_images(
        [img, dst, img_for_drawing],
        ["Original Image", "Harris Corner Response", "Detected Corners"],
        "Task 1: Harris Corner Detection",
        cmap_list=['gray', 'viridis', 'gray']
    )

def task_2_hog_features():
    """Task 2: Extracts HOG features and demonstrates usage."""
    print("Running Task 2: HOG Feature Extraction...")
    img = create_person_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Define HOG Descriptor parameters
    # These must match the image size for this simple case.
    winSize = (64, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9

    # 2. Initialize HOGDescriptor
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    # 3. Compute HOG features
    # The image size must be aligned with winSize.
    resized_img = cv2.resize(gray, winSize)
    hog_features = hog.compute(resized_img)

    print(f"HOG feature vector shape for a {winSize} window: {hog_features.shape}")
    print("This long vector is what a machine learning model would use.")

    # Visualization is complex; we will just show the input image.
    plt.figure(figsize=(5,5))
    plt.imshow(resized_img, cmap='gray')
    plt.title("Input Image for HOG (64x128)")
    plt.suptitle("Task 2: HOG Feature Extraction")
    plt.axis('off')
    plt.show()
    
def task_3_lbp_features():
    """Task 3: Implements a basic LBP operator and shows its histogram."""
    print("Running Task 3: LBP Feature Extraction...")
    img = create_texture_image()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # A simple LBP implementation
    lbp_image = np.zeros_like(gray)
    for y in range(1, gray.shape[0] - 1):
        for x in range(1, gray.shape[1] - 1):
            center = gray[y, x]
            code = 0
            code |= (gray[y-1, x-1] >= center) << 7
            code |= (gray[y-1, x]   >= center) << 6
            code |= (gray[y-1, x+1] >= center) << 5
            code |= (gray[y,   x+1] >= center) << 4
            code |= (gray[y+1, x+1] >= center) << 3
            code |= (gray[y+1, x]   >= center) << 2
            code |= (gray[y+1, x-1] >= center) << 1
            code |= (gray[y,   x-1] >= center) << 0
            lbp_image[y, x] = code

    # Calculate the histogram of LBP codes
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))

    # Display results
    display_images([img, lbp_image], ["Original Texture", "LBP Image"], "Task 3: LBP Feature Extraction")
    
    plt.figure(figsize=(10, 5))
    plt.suptitle("Histogram of LBP Codes")
    plt.bar(range(256), lbp_hist, width=1.0)
    plt.xlabel("LBP Code")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == '__main__':
    task_1_harris_corners()
    task_2_hog_features()
    task_3_lbp_features()

