import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the image size
height, width = 500, 500

# Initialize the image with zeros (black)
image = np.zeros((height, width), dtype=np.uint8)

# Define the vertices of the geometric figure
vertices = np.array([[100, 150], [200, 100], [200, 200], [130, 200]], dtype=np.int32)

# # Reshape vertices for fillPoly
# pts = vertices.reshape((-1, 1, 2))

# Create the mask
cv2.fillPoly(image, [vertices], color=(255))

# Use Harris to detect 4 corners
gray = np.float32(image)
dst = cv2.cornerHarris(gray, 3, 3, 0.04)
corners_indices = np.argpartition(dst.flatten(), -4)[-4:]
x, y = np.unravel_index(corners_indices, dst.shape)
print(x, y)

# Now, 'image' is a mask where the area inside the geometric figure is white (255) and the rest is black (0)
plt.imshow(image, cmap='gray')
plt.scatter(y, x, s=10, c='r')
plt.show()
