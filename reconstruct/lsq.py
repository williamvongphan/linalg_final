import numpy as np
from scipy.sparse import csr_matrix, diags, kron, vstack, hstack, coo_matrix
from scipy.sparse.linalg import spsolve, lsqr, spilu

def splitRGB(image):
    # Split image into RGB channels.
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    # Return list of 3 matrices.
    return [red, green, blue]


def mergeRGB(red, green, blue):
    # Create new image matrix.
    image = np.zeros((red.shape[0], red.shape[1], 3), dtype=np.uint8)
    # Fill image matrix with RGB channels.
    image[:, :, 0] = red
    image[:, :, 1] = green
    image[:, :, 2] = blue
    # Return image matrix.
    return image


def D(n):
    data = [-np.ones(n), np.ones(n - 1)]
    offsets = [0, 1]
    Dn = diags(data, offsets, shape=(n, n)).toarray()
    Dn[n - 1, 0] = 1  # periodic boundary conditions
    Dn = csr_matrix(Dn)
    return Dn

def first_diffs_2d_matrix(m, n):
    In = diags([np.ones(n)], [0])
    Im = diags([np.ones(m)], [0])
    return vstack([kron(In, D(m)), kron(D(n), Im)])


def regularized_ls(A, b, D, delta):
    print(A.shape)
    print(D.shape)
    Atil = vstack([A, delta * csr_matrix(D)])
    btil = np.concatenate((b, np.zeros(D.shape[0])))
    return Atil, btil


def img_to_mask(img_color_channel):
    m, n = img_color_channel.shape
    mask = np.ones((m, n))
    for i in range(m):
        for j in range(n):
            if img_color_channel[i, j] is None:
                mask[i, j] = 0
    return mask


def lsq_reconstruct(json_obj, regParam):
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
    new_img = []
    nparray = np.array(json_obj)
    # Split image into RGB channels.
    red, green, blue = splitRGB(nparray)
    # For each color channel, we will create a regularized least squares matrix and solve for the missing pixels.
    for color in [red, green, blue]:
        """
        y = A * xt;
        λ = 0.3
        Atil, btil =  regularized_ls(A, y, C, λ)
        x_reg_ls = Atil \ Vector(btil)
        @show norm(x_reg_ls - xt)
        Gray.(reshape(x_reg_ls, size(Xt)))
        """
        Xt = color
        delta = 0.3
        m, n = color.shape
        mask = img_to_mask(color)
        mask_true_false_vec = (mask.flatten() == 1)
        # count number of trues in mask
        ntrue = np.sum(mask_true_false_vec)
        A = coo_matrix((np.ones(ntrue), (np.arange(ntrue), np.where(mask_true_false_vec)[0])), shape=(ntrue, m * n)).tocsr()
        xt = Xt.flatten()
        xt[xt is None] = 0
        y = A.astype(np.float64) @ xt.astype(np.float64)
        C = first_diffs_2d_matrix(m, n)
        delta = float(regParam)
        Atil, btil = regularized_ls(A, y, C, delta)
        x_reg_ls = lsqr(Atil, btil, iter_lim=15)[0]
        new_img.append(x_reg_ls.reshape((m, n)).T)
    return mergeRGB(*new_img)