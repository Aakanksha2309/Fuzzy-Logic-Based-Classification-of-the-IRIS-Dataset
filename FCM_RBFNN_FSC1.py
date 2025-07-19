import numpy as np
import matplotlib.pyplot as plt
from graph import simple_fuzzy_plot

def FCM_RBFNN_FSC1(x, f, y, x1, img_name, lambda_reg=0.01, kernel_scale=1.0):
    n, c = f.shape
    n_test, _ = x1.shape
    h = []
    h1 = []
    for i in range(c):
        f_i = f[:, i]
        miu = np.sum(x * f_i[:, np.newaxis], axis=0) / np.sum(f_i)
        centered = x - miu
        sigma = (centered * f_i[:, np.newaxis]).T @ centered / np.sum(f_i)
        if np.linalg.det(sigma) < 1e-6:
            sigma += 1e-2 * np.eye(sigma.shape[0])
        sigma += 1e-4 * np.eye(sigma.shape[0])
        # === TUNING: Scale kernel width for better performance ===
        sigma = sigma * kernel_scale
        sigma_inv = np.linalg.pinv(sigma)
        d_train = np.sum((centered @ sigma_inv) * centered, axis=1)
        h_i = np.exp(-0.5 * d_train)
        centered_test = x1 - miu
        d_test = np.sum((centered_test @ sigma_inv) * centered_test, axis=1)
        h1_i = np.exp(-0.5 * d_test)
        h.append(h_i)
        h1.append(h1_i)
    h = np.stack(h, axis=1)
    h1 = np.stack(h1, axis=1)
    # === TUNING: Adjustable regularization ===
    b = np.linalg.pinv(h.T @ h + lambda_reg * np.eye(h.shape[1])) @ (h.T @ y)
    y_est = h1 @ b
    # === Apply softmax only once ===
    y_est1 = np.exp(y_est) / np.sum(np.exp(y_est), axis=1, keepdims=True)
    return b, y_est1
