import numpy as np

# Create a 2x3 array from a list
my_list = [
    [1, 2, 3],
    [4, 5, 6]
]
print(my_list)
my_array = np.array(my_list)
print(my_array)

# Create an array of zeros
zeros_array = np.zeros((3, 4)) # A 3x4 array
print(zeros_array)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load an image
img = mpimg.imread('/content/imgs/picsum30.jpg')

# Display the image
plt.imshow(img)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 1. Load the image
img = mpimg.imread('/content/imgs/picsum30.jpg')

# 2. Inspect the array
print(f"Image Shape: {img.shape}")
print(f"Data Type: {img.dtype}")

# 3. Display the image
plt.imshow(img)
plt.axis('off') # Hide the axes
plt.show()
