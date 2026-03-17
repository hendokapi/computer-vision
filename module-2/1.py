# We will explore:
# 1. Global vs. Adaptive Thresholding
# 2. Otsu's Automatic Thresholding
# 3. K-Means for Color Quantization
# 4. K-Means for Color-based Object Segmentation

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
        # Check if the image is grayscale or color for correct display
        if len(images[i].shape) == 2:
            plt.imshow(images[i], cmap='gray')
        else:
            # Matplotlib expects RGB, OpenCV uses BGR
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def create_uneven_illumination_image():
    """Creates a sample image with uneven lighting."""
    img = np.zeros((200, 400), dtype=np.uint8)
    # Add some text
    cv2.putText(img, "Text on Left", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (200), 2)
    cv2.putText(img, "Text on Right", (210, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (100), 2)
    # Create a gradient representing uneven light
    gradient = np.linspace(1.0, 0.2, 400)
    img = np.uint8(img * gradient)
    return img

def create_bimodal_image():
    """Creates a sample image with two distinct intensity peaks."""
    img = np.full((200, 200), 220, dtype=np.uint8) # Light background
    cv2.circle(img, (100, 100), 50, 50, -1) # Dark circle
    cv2.add(img, np.random.randint(-10, 11, img.shape, dtype=np.int8), img, dtype=cv2.CV_8U) # Add noise
    return img

def create_colorful_image():
    """Creates a sample image with various colors."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Different colored rectangles
    img[0:100, 0:100] = [255, 0, 0]       # Red
    img[0:100, 100:200] = [0, 255, 0]     # Green
    img[100:200, 0:100] = [0, 0, 255]     # Blue
    img[100:200, 100:200] = [255, 255, 0] # Yellow
    return img

def create_colored_objects_image():
    """Creates an image with distinct colored objects on a white background."""
    img = np.full((200, 200, 3), 255, dtype=np.uint8) # White background
    cv2.circle(img, (50, 50), 30, (200, 20, 20), -1)  # Red circle
    cv2.rectangle(img, (120, 120), (180, 180), (20, 200, 20), -1) # Green rectangle
    return img

# --- Main Lab Tasks ---

def task_1_thresholding_comparison():
    """
    Task 1: Demonstrates the failure of global thresholding and the success
    of adaptive thresholding on an image with uneven illumination.
    """
    print("Running Task 1: Global vs. Adaptive Thresholding...")
    # 1. Create and load the image
    gray_img = create_uneven_illumination_image()

    # 2. Apply Global Thresholding (manual value)
    _, global_thresh = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY)

    # 3. Apply Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4
    )

    # 4. Display results
    display_images(
        [gray_img, global_thresh, adaptive_thresh],
        ["Original (Uneven Light)", "Global Threshold (T=100)", "Adaptive Threshold"],
        "Task 1: Global vs. Adaptive Thresholding"
    )

def task_2_otsu_method():
    """
    Task 2: Applies Otsu's method to automatically find the optimal
    threshold for a bimodal image.
    """
    print("Running Task 2: Otsu's Automatic Thresholding...")
    # 1. Create and load the image
    gray_img = create_bimodal_image()

    # 2. Use cv2.threshold with the cv2.THRESH_OTSU flag
    # The threshold value is automatically calculated and returned
    otsu_threshold, otsu_thresh_img = cv2.threshold(gray_img, 0, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Print the optimal threshold value found by Otsu's method
    print(f"Otsu's method found an optimal threshold of: {otsu_threshold}")

    # 4. Display results
    display_images(
        [gray_img, otsu_thresh_img],
        ["Original Bimodal Image", f"Otsu's Threshold (T={int(otsu_threshold)})"],
        "Task 2: Automatic Thresholding with Otsu's Method"
    )


def task_3_color_quantization():
    """
    Task 3: Uses K-Means to reduce the number of colors in an image, a process
    known as color quantization.
    """
    print("Running Task 3: Color Quantization with K-Means...")
    # 1. Load the color image
    img = create_colorful_image()

    # 2. Reshape the image data to be a list of pixels (N_pixels x 3)
    pixel_vals = img.reshape((-1, 3))
    # Convert to float32, as required by cv2.kmeans
    pixel_vals = np.float32(pixel_vals)

    # 3. Define stopping criteria and run cv2.kmeans()
    # Stop after 100 iterations or if the centers move by less than 0.2 pixels
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = 2 # Number of clusters (colors) we want to find
    retval, labels, centers = cv2.kmeans(
        pixel_vals,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    # 4. Reconstruct the image from the K-Means results
    # Convert cluster centers back to uint8
    center_colors = np.uint8(centers)
    # Map each pixel's label to the corresponding center color
    segmented_data = center_colors[labels.flatten()]
    # Reshape the data back into the original image dimensions
    segmented_image = segmented_data.reshape((img.shape))

    # 5. Display results
    display_images(
        [img, segmented_image],
        ["Original Colorful Image", f"Quantized to K={K} Colors"],
        "Task 3: Color Quantization with K-Means"
    )

def task_4_color_segmentation():
    """
    Task 4: Uses K-Means to separate objects based on color, creating a
    binary mask for each object.
    """
    print("Running Task 4: Color-based Segmentation with K-Means...")
    img = create_colored_objects_image()

    # K-Means process (same as Task 3)
    pixel_vals = img.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = 3 # 3 clusters: background, object 1, object 2
    retval, labels, centers = cv2.kmeans(
        pixel_vals,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )
    
    # The background is white (255, 255, 255). Find which cluster center is the background.
    # The cluster center with the highest sum of RGB values is likely the white background.
    background_cluster_index = np.argmax(np.sum(centers, axis=1))

    # Create masks for each foreground object
    # Find all pixels that are NOT the background cluster
    masks = []
    titles = []
    
    label_matrix = labels.reshape(img.shape[:2])

    for i in range(K):
        if i != background_cluster_index:
            # Create a mask where pixels belonging to cluster 'i' are white (255)
            # and all others are black (0).
            mask = np.uint8(label_matrix == i) * 255
            masks.append(mask)
            # Find the average color of the cluster for the title
            color = np.uint8(centers[i])
            titles.append(f"Mask for Cluster {i} (Color: {color})")
            
    display_images(
        [img] + masks,
        ["Original"] + titles,
        "Task 4: Isolating Objects with Color-based Segmentation"
    )


if __name__ == '__main__':
    # Run all tasks sequentially
    task_1_thresholding_comparison()
    task_2_otsu_method()
    task_3_color_quantization()
    task_4_color_segmentation()
