import os
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from glob import glob

from config import params

# Path to the directory containing the images
images_path = params.images_path

# Get images paths
images = sorted(glob(os.path.join(images_path, "*")))

# Loop for flipping images
for image in images:
    # Read image
    img = mpimg.imread(image)
    # Flip image
    flipped_image = cv2.flip(img, 1)
    # Save flipped image
    mpimg.imsave(image, flipped_image)