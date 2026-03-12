import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def show_image(title, image, cmap=None):
    plt.figure(figsize=(5,4))
    if cmap:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

# ------------------------------------------------------------
# Create Binary Test Image
# ------------------------------------------------------------

def create_binary_test_image():
    img = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), 255, -1)
    cv2.circle(img, (110, 125), 25, 0, -1)
    cv2.circle(img, (170, 90), 3, 0, -1) 
    cv2.circle(img, (300, 100), 50, 255, -1)
    cv2.putText(img, "CV", (120, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 5)

    # --- Add salt noise ---
    noise_coords = (
        np.random.randint(0, 400, 800),
        np.random.randint(0, 400, 800)
    )
    img[noise_coords] = 255

    return img

# ------------------------------------------------------------
# Fundamental Morphological Operations
# ------------------------------------------------------------

def basic_morphology(image):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    erosion = cv2.erode(image, kernel, iterations=1)
    dilation = cv2.dilate(image, kernel, iterations=1)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return erosion, dilation, opening, closing

# ------------------------------------------------------------
# Advanced Morphological Operations
# ------------------------------------------------------------

def advanced_morphology(image):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

    gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    return gradient, top_hat, black_hat * 255

# ------------------------------------------------------------
# Laboratory: Real Use Cases
# ------------------------------------------------------------

def morphology_lab():

    image = create_binary_test_image()
    show_image("Original Binary Image (with Noise)", image, cmap='gray')

    # Basic Operations
    erosion, dilation, opening, closing = basic_morphology(image)

    show_image("Erosion", erosion, cmap='gray')
    show_image("Dilation", dilation, cmap='gray')

    # erosion -> dilation
    # Removes small bright noise
    # Preserves object size (mostly)
    # Smooths contours
    show_image("Opening (Noise Removal)", opening, cmap='gray')

    # dilation -> opening
    # Fills small holes
    # Connects nearby regions
    # Smooths boundaries
    show_image("Closing (Fill Gaps)", closing, cmap='gray')

    # Advanced
    gradient, top_hat, black_hat = advanced_morphology(image)

    # dilation - erosion
    # extracts objects boundaries
    show_image("Morphological Gradient (Edge Extraction)", gradient, cmap='gray')

    # image - opening
    # image - (erosion -> dilation)
    # extracts small bright regions
    show_image("Top-Hat (Small Bright Objects)", top_hat, cmap='gray')

    # closing - image
    # (dilation -> opening) - image
    # extracts small dark regions
    show_image("Black-Hat (Dark Regions)", black_hat, cmap='gray')

if __name__ == "__main__":
    morphology_lab()
