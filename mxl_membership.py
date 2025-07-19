import numpy as np

def mxl_membership(x, mean, varcov):
    m, n = x.shape
    x1 = np.zeros_like(x)

    for i in range(n):
        x1[:, i] = x[:, i] - mean[i]

    P1 = x1 @ np.linalg.inv(varcov)
    P2 = P1 * x1
    P = np.sum(P2, axis=1)

    return P
