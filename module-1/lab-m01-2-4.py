import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load image
    image = cv2.imread("../images/poor_quality_scene.jpg")

    if image is None:
        print("Error loading image")
        return

    analyze(image)

    wb_image = gray_world_white_balance(image)
    analyze(wb_image)
    show_results("white balanced", image, wb_image)

    denoised = cv2.medianBlur(wb_image, 5)
    show_results("denoised", image, denoised)

    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v_eq = cv2.equalizeHist(v)

    hsv_eq = cv2.merge([h, s, v_eq])
    enhanced = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    show_results("Original vs Processed", image, enhanced)

    cv2.imwrite("../results/processed_result.png", enhanced)
    print("Processing complete. Saved as processed_result.png")

def analyze(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    colors = ('b', 'g', 'r')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
    plt.title("Histogram (BGR)")
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()

def show_results(title, a, b):
    a_rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    b_rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)

    combined = np.hstack((a_rgb, b_rgb))

    plt.figure(figsize=(14, 6))
    plt.imshow(combined)
    plt.title(title)
    plt.axis("off")
    plt.show()

def gray_world_white_balance(img):
    img = img.astype(np.float32)

    # Compute average per channel
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])

    # Compute global average
    avg_gray = (avg_b + avg_g + avg_r) / 3

    # Scale channels
    img[:, :, 0] *= (avg_gray / avg_b)
    img[:, :, 1] *= (avg_gray / avg_g)
    img[:, :, 2] *= (avg_gray / avg_r)

    # Clip values to valid range
    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)

if __name__ == "__main__":
    main()
