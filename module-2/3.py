# This script provides a hands-on demonstration of the Marker-Controlled Watershed
# algorithm to solve the common problem of separating touching objects.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Helper Functions ---

def display_images(images, titles, main_title):
    """Helper to display multiple images with titles."""
    num_images = len(images)
    # Increase figure size for better viewing of multiple steps
    plt.figure(figsize=(4 * num_images, 4))
    plt.suptitle(main_title, fontsize=16)
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        # Use 'viridis' colormap for single-channel images to better see intensity differences
        cmap = 'viridis' if len(images[i].shape) == 2 else None
        if len(images[i].shape) == 2:
            plt.imshow(images[i], cmap=cmap)
        else:
            # Convert BGR to RGB for correct color display with Matplotlib
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def create_touching_coins_image():
    """Creates an image with 10 touching, but not overlapping, coins."""
    # Create a dark gray background
    img = np.full((300, 500, 3), 40, dtype=np.uint8)

    # Define positions and radii for 10 coins, arranged to touch
    coin_specs = [
        # (center_x, center_y), radius
        ((70, 70), 30), 
        ((135, 90), 28), 
        ((200, 65), 32), 
        ((270, 80), 35),
        ((350, 75), 40),
        ((60, 160), 35), 
        ((130, 170), 40), 
        ((220, 150), 30), 
        ((300, 180), 38),
        ((400, 160), 36),
    ]

    for i, (center, radius) in enumerate(coin_specs):
        # Create a slightly different shade of gray/silver for each coin
        color_val = 150 + i * 5
        color = (color_val, color_val, color_val)
        cv2.circle(img, center, radius, color, -1)
        # Add a bit of noise to each coin to make it look less perfect
        noise = np.zeros((radius*2, radius*2, 3), dtype=np.int8)
        cv2.randn(noise, 0, 5)
        # Apply noise only within the circle area
        roi = img[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius]
        mask = np.zeros((radius*2, radius*2), dtype=np.uint8)
        cv2.circle(mask, (radius, radius), radius, 255, -1)
        # Use mask to add noise only to the coin
        roi_noisy = cv2.add(roi, noise, dst=roi, mask=mask, dtype=cv2.CV_8UC3)

    return img

# --- Main Lab Task ---

def watershed_segmentation_pipeline():
    """
    Implements the full Marker-Controlled Watershed pipeline to separate
    the touching coins.
    """
    print("Running Marker-Controlled Watershed Pipeline...")
    img = create_touching_coins_image()

    # --- Step 1: Pre-processing and create a Binary Image ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Using Otsu's method to automatically find the optimal threshold
    # We use THRESH_BINARY because our objects are brighter than the background
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Step 2: Noise Removal ---
    # Morphological Opening (erosion followed by dilation) removes small noise
    # that might be present in the background.
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- Step 3: Finding Sure Background Area ---
    # Dilating the objects makes the remaining background area smaller,
    # ensuring it is truly the background. This is our first marker.
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # --- Step 4: Finding Sure Foreground Area ---
    # The Distance Transform finds the distance of each pixel from the nearest
    # zero-valued pixel (the background). The centers of the coins will be brightest.
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # Threshold the distance map to get the central cores of the coins.
    # This gives us our second set of markers.
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # --- Step 5: Finding the Unknown Region ---
    # This is the ambiguous region between the sure foreground and sure background.
    # The watershed algorithm will decide where the boundaries go in this region.
    unknown = cv2.subtract(sure_bg, sure_fg)

    # --- Step 6: Creating the Marker Image for Watershed ---
    # Label the sure foreground objects with unique positive integers (starting from 2).
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add 1 to all labels so that the sure background (originally 0) is not 0, but 1.
    markers = markers + 1
    # Mark the unknown region with 0. This is what the watershed algorithm will fill.
    markers[unknown == 255] = 0

    # --- Step 7: Apply the Watershed Algorithm ---
    # The algorithm modifies our markers image in-place.
    # The boundaries between regions will be marked with -1.
    cv2.watershed(img, markers)

    # --- Step 8: Visualize the Result ---
    # We want to draw the boundary lines on the original image.
    img_result = img.copy()
    img_result[markers == -1] = [0, 0, 255] # Draw boundaries in Red
    
    # For a clearer view of just the markers
    vis_markers_color = cv2.applyColorMap(np.uint8(markers * 10), cv2.COLORMAP_JET)
    vis_markers_color[markers == 0] = [0, 0, 0] # Unknown region in Black
    vis_markers_color[markers == -1] = [0, 0, 255] # Boundaries in Red

    # Display all the intermediate steps to understand the process
    display_images(
        [img, thresh, sure_bg, dist_transform, sure_fg, vis_markers_color, img_result],
        ["Original", "Threshold", "Sure BG", "Distance Tx", "Sure FG", "Markers", "Result"],
        "Watershed Algorithm Pipeline"
    )

if __name__ == '__main__':
    watershed_segmentation_pipeline()
