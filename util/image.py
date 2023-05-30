# Image manipulation utilities.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as smp
import math
import sys


# Split image into RGB channels, return as list of 3 matrices.
def splitRGB(image):
    # Split image into RGB channels.
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    # Return list of 3 matrices.
    return [red, green, blue]

# your jokes are so funny :rolling_eyes: :sideeye:


# Convert image to grayscale.
def grayscale(image):
    # Split image into RGB channels.
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    # Convert to grayscale.
    gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue
    # Return grayscale image.
    return gray


# Convert image from 3 matrices to 1 matrix.
def mergeRGB(red, green, blue):
    # Create new image matrix.
    image = np.zeros((red.shape[0], red.shape[1], 3), dtype=np.uint8)
    # Fill image matrix with RGB channels.
    image[:, :, 0] = red
    image[:, :, 1] = green
    image[:, :, 2] = blue
    # Return image matrix.
    return image
