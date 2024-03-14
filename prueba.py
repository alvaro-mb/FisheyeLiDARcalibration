import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the image size
height, width = 500, 500

# Initialize the image with zeros (black)
image = np.zeros((height, width), dtype=np.uint8)

# Define the vertices of the geometric figure
vertices = np.array([[100, 150], [200, 100], [200, 200], [130, 200]], dtype=np.int32)

# Reshape vertices for fillPoly
pts = vertices.reshape((-1, 1, 2))

# Create the mask
cv2.fillPoly(image, [pts], color=(255))

# Now, 'image' is a mask where the area inside the geometric figure is white (255) and the rest is black (0)
plt.imshow(image, cmap='gray')
plt.show()
