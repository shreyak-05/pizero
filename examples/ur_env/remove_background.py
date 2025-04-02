import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = '/content/image_23_52_38_072.png'
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Define the blue color range in HSV
lower_blue = np.array([100, 150, 50])  # Lower bound of blue
upper_blue = np.array([140, 255, 255])  # Upper bound of blue
# Create a mask for the blue color
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND the mask and the original image to keep only the blue parts
result = cv2.bitwise_and(image, image, mask=mask)
# Convert the result to an image with transparent background where mask is not present
result_with_alpha = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)  # Convert to 4 channels (BGR + Alpha)
# Set alpha to 0 for non-blue areas (where mask is 0)
result_with_alpha[:, :, 3] = mask
# Display the image
plt.imshow(cv2.cvtColor(result_with_alpha, cv2.COLOR_BGRA2RGBA))
plt.axis('off') 
plt.show()
# Save the output image with transparency
cv2.imwrite('./blue_sketch_with_transparency.png', result_with_alpha)