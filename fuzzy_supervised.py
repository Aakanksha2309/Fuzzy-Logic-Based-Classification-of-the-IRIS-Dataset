
import numpy as np
import matplotlib.pyplot as plt

def fuzzysupervised(f, x, plot_results=False, img_prefix=None):
    m, n = x.shape
    z = np.zeros((m, n))
    for i in range(n):
        z[:, i] = x[:, i] * f
    k = np.sum(f)
    z1 = np.sum(z, axis=0)
    fmean = np.zeros(n)
    for i in range(n):
        fmean[i] = z1[i] / k
        x[:, i] = x[:, i] - fmean[i]
    x1 = x ** 2
    for i in range(n):
        z[:, i] = x1[:, i] * f
    z2 = np.sum(z, axis=0)
    fvar = np.zeros(n)
    for i in range(n):
        fvar[i] = z2[i] / k
    k1 = 0
    cross = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            cross_col = (x[:, i] * x[:, j]) * f
            cross.append(cross_col)
            k1 += 1
    cross = np.array(cross)
    cvar = np.sum(cross, axis=1) / k
    c = np.zeros((n, n))
    k2 = 0
    for i in range(n):
        c[i, i] = fvar[i]
        for j in range(i + 1, n):
            c[i, j] = cvar[k2]
            c[j, i] = c[i, j]
            k2 += 1
    # ===== IMPROVED: Add regularization for numerical stability =====
    c += 1e-4 * np.eye(n)
    eigenvals, eigenvecs = np.linalg.eigh(c)
    eigenvals = np.maximum(eigenvals, 1e-6)
    c = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    if plot_results and img_prefix:
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        for i in range(n):
            plt.hist(x[:, i], bins=20, alpha=0.5, label=f'Feature {i+1}')
        plt.title('Original Feature Distributions')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.subplot(132)
        for i in range(n):
            plt.hist(z[:, i], bins=20, alpha=0.5, label=f'Feature {i+1}')
        plt.title('Fuzzy-Weighted Distributions')
        plt.xlabel('Weighted Value')
        plt.legend()
        
        # Plot 3: Covariance Matrix Heatmap
        plt.subplot(133)
        plt.imshow(c, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Covariance')
        plt.title('Fuzzy Covariance Matrix')
        plt.xticks(np.arange(n), labels=np.arange(1, n+1))
        plt.yticks(np.arange(n), labels=np.arange(1, n+1))
        
        plt.tight_layout()
        plt.savefig(f'results/{img_prefix}_fuzzy_supervised.jpeg', dpi=300)
        plt.close()

    return fmean, c
