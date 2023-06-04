import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, hstack, vstack, csr_matrix, diags, kron
from scipy.sparse.linalg import spsolve, lsqr, spilu
import cvxpy as cp


def splitRGB(image):
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    return [red, green, blue]

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


def mergeRGB(red, green, blue):
    image = np.zeros((red.shape[0], red.shape[1], 3), dtype=np.uint8)
    image[:, :, 0] = red
    image[:, :, 1] = green
    image[:, :, 2] = blue
    return image

"""
def lasso_admm(y, A, C, lambda_val, rho_val, nIters):
    P = A.T * A + rho_val * (C.T * C)
    Aty = A.T * y
    x = np.zeros(A.shape[1])
    z = np.zeros(C.shape[0])
    u = np.zeros(C.shape[0])
    print("P shape: ", P.shape)
    print("Aty shape: ", Aty.shape)
    print("A shape: ", A.shape)
    print("C shape: ", C.shape)
    print("x shape: ", x.shape)
    print("z shape: ", z.shape)
    print("u shape: ", u.shape)
    for i in range(nIters):
        print("iteration: ", i)
        x = lsqr(P, Aty + rho_val * (C.T @ (z - u)), iter_lim=2)[0]
        Cx = C * x
        z = np.sign(Cx + u) * np.maximum(np.abs(Cx + u) - lambda_val / 2, 0)
        u = u + Cx - z
    return x
"""

def lasso_cvxpy (y, A, C, lambda_val):
    x = cp.Variable(A.shape[1])
    objective = cp.Minimize(cp.sum_squares(A @ x - y) + lambda_val * cp.norm(C @ x, 1))
    prob = cp.Problem(objective)
    prob.solve(solver=cp.SCS, verbose=False, max_iters=1000, eps=1e-3)
    return x.value

def img_to_mask(img_color_channel):
    m, n = img_color_channel.shape
    mask = np.ones((m, n))
    for i in range(m):
        for j in range(n):
            if img_color_channel[i, j] is None:
                mask[i, j] = 0
    return mask


def lasso_reconstruct(json_obj, alpha=0.05):
    new_img = []
    nparray = np.array(json_obj)
    # Split image into RGB channels.
    red, green, blue = splitRGB(nparray)
    # For each color channel, we will create a regularized least squares matrix and solve for the missing pixels.
    for color in [red, green, blue]:
        Xt = color
        m, n = color.shape
        mask = img_to_mask(color)
        mask_true_false_vec = (mask.flatten() == 1)
        # count number of trues in mask
        ntrue = np.sum(mask_true_false_vec)
        A = coo_matrix((np.ones(ntrue), (np.arange(ntrue), np.where(mask_true_false_vec)[0])),
                       shape=(ntrue, m * n)).tocsr() # what if you just hack on his acc and get him banned fr
        xt = Xt.flatten()
        xt[xt is None] = 0
        y = A.astype(np.float64) @ xt.astype(np.float64)
        print(y[0:10])
        C = first_diffs_2d_matrix(m, n)
        alpha = float(alpha)
        res = lasso_cvxpy(y, A, C, alpha)
        new_img.append(res.reshape((m, n)).T)
    return mergeRGB(*new_img)