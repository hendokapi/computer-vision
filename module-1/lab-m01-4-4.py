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


def show_spectrum(title, magnitude_spectrum):
    plt.figure(figsize=(5, 4))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(title)
    plt.axis("off")
    plt.show()

# ------------------------------------------------------------
# Generate Test Image with Illumination Problem + Noise
# ------------------------------------------------------------

def create_test_image():
    img = np.zeros((512, 512), dtype=np.float32)

    # Gradient illumination
    for i in range(512):
        img[:, i] = 50 + (200 * (i / 512))

    # Add object
    cv2.circle(img, (256, 256), 80, 255, -1)

    # Add Gaussian noise
    noise = np.random.normal(0, 20, img.shape)
    img += noise

    return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# Fourier Transform
# ------------------------------------------------------------

def compute_fft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    return fshift, magnitude


def inverse_fft(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)


# ------------------------------------------------------------
# Frequency Filters
# ------------------------------------------------------------

def create_low_pass_mask(shape, radius=40):
    rows, cols = shape
    mask = np.zeros((rows, cols), np.uint8)
    crow, ccol = rows // 2, cols // 2

    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow)**2 + (j - ccol)**2) <= radius:
                mask[i, j] = 1
    return mask


def create_high_pass_mask(shape, radius=40):
    return 1 - create_low_pass_mask(shape, radius)


def create_band_pass_mask(shape, r_inner=20, r_outer=60):
    rows, cols = shape
    mask = np.zeros((rows, cols), np.uint8)
    crow, ccol = rows // 2, cols // 2

    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if r_inner <= d <= r_outer:
                mask[i, j] = 1
    return mask

def apply_frequency_filter(image, mask):
    fshift, magnitude = compute_fft(image)
    filtered = fshift * mask
    result = inverse_fft(filtered)
    return np.clip(result, 0, 255).astype(np.uint8)

# ------------------------------------------------------------
# Execution
# ------------------------------------------------------------

if __name__ == "__main__":

    image = create_test_image()
    show_image("Original Image (Illumination + Noise)", image, cmap='gray')

    # FFT
    fshift, magnitude = compute_fft(image)
    show_spectrum("Magnitude Spectrum (Log Scale)", magnitude)

    # Frequency Filters
    low_pass = apply_frequency_filter(image, create_low_pass_mask(image.shape))
    high_pass = apply_frequency_filter(image, create_high_pass_mask(image.shape))
    band_pass = apply_frequency_filter(image, create_band_pass_mask(image.shape))

    show_image("Low-Pass Filter Result", low_pass, cmap='gray')
    show_image("High-Pass Filter Result", high_pass, cmap='gray')
    show_image("Band-Pass Filter Result", band_pass, cmap='gray')
