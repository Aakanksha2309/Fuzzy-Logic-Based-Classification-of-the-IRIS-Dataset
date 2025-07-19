import numpy as np
import matplotlib.pyplot as plt
from mxl_membership import mxl_membership
from fuzzy_supervised import fuzzysupervised


def fsc_generation(sat_data, training_sample, no_class, img_name):
    x = sat_data
    x1 = training_sample
    m0, n0 = x.shape
    m, n = x1.shape
    u1 = []
    P1 = []

    for i in range(no_class):
        fmean_class, fvarcov_class = fuzzysupervised(x1[:, n0 + i], x1[:, :n0])
        P_class = mxl_membership(x[:, :n0], fmean_class, fvarcov_class)
        N = n0
        constant = (1 / ((2 * np.pi) ** (N / 2))) * (np.sqrt(np.linalg.det(fvarcov_class)))
        P = constant * np.exp(-0.5 * P_class)
        P1.append(P)

    P1 = np.column_stack(P1)
    P2 = np.sum(P1, axis=1, keepdims=True)
    u1 = P1 / P2

    c_fsc = np.max(u1, axis=1)
    I_fsc = np.argmax(u1, axis=1) + 1
    I_fsc = I_fsc.reshape(15, 10)

    # ========== PLOTTING REMOVED ==========
    # Simple cluster plots will be generated in main.py
    # =====================================

    return u1

