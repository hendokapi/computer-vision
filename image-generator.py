import cv2
import numpy as np
import os

# ------------------------------------------------------------
# Create output folder
# ------------------------------------------------------------
output_dir = "synthetic_images"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------
# Utility to save images
# ------------------------------------------------------------
def save_image(image, filename):
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, image)
    print(f"Saved: {path}")

# ------------------------------------------------------------
# 1️⃣ Create clean synthetic scene
# ------------------------------------------------------------
def create_clean_scene(width=512, height=512):
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Gradient background
    for i in range(width):
        intensity = int(255 * (i / width))
        image[:, i] = (intensity, intensity, intensity)

    # Add colored shapes
    cv2.circle(image, (150, 150), 60, (0, 0, 255), -1)      # Red circle
    cv2.rectangle(image, (300, 100), (450, 250), (0, 255, 0), -1)  # Green rectangle
    cv2.line(image, (50, 400), (450, 350), (255, 0, 0), 5)  # Blue line

    cv2.putText(image, "VISION LAB", (120, 480),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    return image

# ------------------------------------------------------------
# 2️⃣ Degradation Models
# ------------------------------------------------------------
def apply_underexposure(image, factor=0.5):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def apply_color_cast(image, cast=(1.3, 1.0, 0.8)):
    img = image.astype(np.float32)
    img[:, :, 0] *= cast[0]
    img[:, :, 1] *= cast[1]
    img[:, :, 2] *= cast[2]
    return np.clip(img, 0, 255).astype(np.uint8)

def add_gaussian_noise(image, sigma=30):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, amount=0.01):
    noisy = image.copy()
    num_pixels = int(amount * image.shape[0] * image.shape[1])

    # Salt
    coords = (
        np.random.randint(0, image.shape[0], num_pixels),
        np.random.randint(0, image.shape[1], num_pixels)
    )
    noisy[coords] = 255

    # Pepper
    coords = (
        np.random.randint(0, image.shape[0], num_pixels),
        np.random.randint(0, image.shape[1], num_pixels)
    )
    noisy[coords] = 0

    return noisy

def apply_blur(image, ksize=7):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def compress_dynamic_range(image):
    img = image.astype(np.float32)
    img = img / 255.0
    img = np.power(img, 2.2)
    img = img * 255
    return np.clip(img, 0, 255).astype(np.uint8)

# ------------------------------------------------------------
# 3️⃣ Full Poor-Quality Generator
# ------------------------------------------------------------
def generate_poor_quality_image():
    clean = create_clean_scene()
    degraded = clean.copy()

    degraded = apply_underexposure(degraded, factor=0.5)
    degraded = apply_color_cast(degraded, cast=(1.4, 1.0, 0.7))
    degraded = add_gaussian_noise(degraded, sigma=30)
    degraded = add_salt_pepper_noise(degraded, amount=0.01)
    degraded = apply_blur(degraded, ksize=7)
    degraded = compress_dynamic_range(degraded)

    return clean, degraded

# ------------------------------------------------------------
# 4️⃣ Run Generator and Save Images
# ------------------------------------------------------------
if __name__ == "__main__":
    clean, poor = generate_poor_quality_image()

    # Save images
    save_image(clean, "clean_scene.jpg")
    save_image(poor, "poor_quality_scene.jpg")
