import cv2
# from google.colab.patches import cv2_imshow

def show_image(title, img):
  # Display the image in a window named "My Image"
  cv2.imshow(title, img)
  # Wait for the user to press any key
  cv2.waitKey(0)
  # Clean up
  cv2.destroyAllWindows()

# Load an image from disk
img = cv2.imread('/content/imgs/picsum30.jpg')

# Check if the image was loaded correctly
if img is None:
  print("Error: Could not read the image.")
else:
  print(f"Image loaded with shape: {img.shape}")

# cv2_imshow(img) # on Colab use this
show_image('Image 1', img)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load an image
# img = mpimg.imread('/content/imgs/picsum30.jpg')

# Display the image
plt.imshow(img)
plt.show()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()

height, width = img.shape[:2]      # Get original dimensions

# Resize to a fixed size of 300x200
resized_img = cv2.resize(img, (300, 100))
show_image('Image 2', resized_img)

resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
plt.imshow(resized_img)
plt.show()

# Resize by a scaling factor (e.g., 50% smaller)
scaled_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
show_image('Image 3', scaled_img)

scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)
plt.imshow(scaled_img)
plt.show()

# Convert the BGR image to Grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Check the new shape
print(f"Grayscale image shape: {gray_img.shape}")
# Note: It's now 2D!

gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)
plt.imshow(gray_img)
plt.show()
