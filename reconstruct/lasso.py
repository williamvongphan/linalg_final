import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from scipy.sparse import coo_matrix, hstack, vstack


def splitRGB(image):
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    return [red, green, blue]


def mergeRGB(red, green, blue):
    image = np.zeros((red.shape[0], red.shape[1], 3), dtype=np.uint8)
    image[:, :, 0] = red
    image[:, :, 1] = green
    image[:, :, 2] = blue
    return image


def solve_lasso(A, b, alpha):
    lasso = Lasso(alpha=alpha, fit_intercept=False, tol=0.01, max_iter=10000)
    print(A[0:10, 0:10])
    print(b[0:10])
    lasso.fit(A, b)
    print("Lasso coefficients: ")
    print(lasso.coef_)
    return lasso.coef_

def img_to_mask(img_color_channel):
    m, n = img_color_channel.shape
    mask = np.ones((m, n))
    for i in range(m):
        for j in range(n):
            if img_color_channel[i, j] is None:
                mask[i, j] = 0
    return mask



def reconstruct_image(image, mask, alpha=0.1):
    rows, cols, _ = image.shape
    channels = splitRGB(image)
    print("Image shape")
    print(image.shape)

    mask_flat = mask.flatten()
    missing_pixels = np.where(mask_flat == 0)[0]
    known_pixels = np.where(mask_flat == 1)[0]

    A = coo_matrix((rows * cols, rows * cols))
    # Count number of missing pixels
    print(str(len(missing_pixels)) + " missing pixels")
    print(A.shape)
    print(str(len(list(zip(*np.where(mask == 0))))) + " iterations")
    for i, j in zip(*np.where(mask == 0)):
        row_idx = i * cols + j
        A = A + coo_matrix(([1], ([row_idx], [row_idx])), shape=(rows * cols, rows * cols))
    reconstructed_channels = []
    print("Reconstructing channels...")
    for channel in channels:
        # For all channels, replace None with 0
        for i in range(rows):
            for j in range(cols):
                if channel[i, j] is None:
                    channel[i, j] = 0
        b = channel.flatten()
        b_known = b[known_pixels]

        x = solve_lasso(A, b, alpha=alpha)

        reconstructed = np.zeros(rows * cols)
        reconstructed[missing_pixels] = x[missing_pixels]
        reconstructed[known_pixels] = b_known

        reconstructed_channel = np.reshape(reconstructed, (rows, cols)).clip(0, 255).astype(np.uint8)
        reconstructed_channels.append(reconstructed_channel.T)

    return mergeRGB(*reconstructed_channels)


def lasso_reconstruct(json_obj, alpha=0.1):
    """
    json_obj is always in this format:
    [
      [
        [red, green, blue],
        [red, green, blue],
        ...
      ],
      [
        [red, green, blue],
        [red, green, blue],
        ...
      ],
      ...
    ]
    Sometimes an array can contain [None, None, None], this means that the pixel was removed and we aim to reconstruct it.
    :param json_obj:
    :return:
    """
    nparray = np.array(json_obj)
    # Get mask
    mask = img_to_mask(nparray[:, :, 0])
    print("Mask shape: ", mask.shape)
    # Pass all to reconstruct_image
    new_img = reconstruct_image(nparray, mask, alpha=float(alpha))
    return new_img