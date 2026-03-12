import cv2
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def show_image(title, image, cmap=None):
    plt.figure(figsize=(5, 4))
    if cmap:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

# ------------------------------------------------------------
# Generate Test Image with Noise
# ------------------------------------------------------------

def create_test_image():
    img = np.zeros((512, 512, 3), dtype=np.uint8)

    # Gradient background
    for i in range(512):
        intensity = int(255 * (i / 512))
        img[:, i] = (intensity, intensity, intensity)

    cv2.circle(img, (150, 150), 70, (0, 0, 255), -1)
    cv2.rectangle(img, (300, 100), (450, 250), (0, 255, 0), -1)
    cv2.putText(img, "FILTER LAB", (100, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    return img


def add_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# Convolution vs Correlation
# ------------------------------------------------------------

def correlation(image, kernel):
    return cv2.filter2D(image, -1, kernel)


def convolution(image, kernel):
    flipped_kernel = np.flip(kernel)
    return cv2.filter2D(image, -1, flipped_kernel)


# ------------------------------------------------------------
# Linear Filters
# ------------------------------------------------------------

def averaging_filter(image):
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(image, -1, kernel)


def gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 1.5)


def sharpening_filter(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


# ------------------------------------------------------------
# Non-Linear Filters
# ------------------------------------------------------------

def median_filter(image):
    return cv2.medianBlur(image, 5)


def bilateral_filter(image):
    return cv2.bilateralFilter(image, 9, 75, 75)


# ------------------------------------------------------------
# Execution
# ------------------------------------------------------------

if __name__ == "__main__":

    clean = create_test_image()
    noisy = add_noise(clean)

    show_image("Clean Image", clean)
    show_image("Noisy Image", noisy)

    results = {}

    filters = {
        "Averaging": averaging_filter,
        "Gaussian": gaussian_filter,
        "Sharpening": sharpening_filter,
        "Median (Non-Linear)": median_filter,
        "Bilateral (Non-Linear)": bilateral_filter
    }

    for name, func in filters.items():
        output = func(noisy)
        show_image(f"{name} Output", output)

