# import numpy as np

# def FEM(u3, u4):
#     """
#     Compares two fuzzy membership matrices.
#     Returns: Confusion Matrix, OA, Kappa, MSE, Cross Entropy
#     """
#     if u3.shape != u4.shape:
#         raise ValueError(f"Shape mismatch: {u3.shape} vs {u4.shape}")

#     # Normalize and sanitize
#     u3 = u3 / np.sum(u3, axis=1, keepdims=True)
#     u4 = u4 / np.sum(u4, axis=1, keepdims=True)
#     u3 = np.nan_to_num(u3)
#     u4 = np.nan_to_num(u4)

#     c = u3.shape[1]
#     E = np.zeros((c, c))
#     for i in range(c):
#         for j in range(c):
#             E[i, j] = np.sum(np.minimum(u3[:, i], u4[:, j]))

#     row_sums = np.sum(E, axis=1)
#     col_sums = np.sum(E, axis=0)
#     total = np.sum(E)
#     diag = np.diag(E)

#     OA = 100 * np.sum(diag) / total if total != 0 else 0

#     expected_accuracy = np.dot(row_sums, col_sums) / (total ** 2)
#     observed_accuracy = np.sum(diag) / total
#     kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy) if (1 - expected_accuracy) != 0 else 0

#     mse = np.mean(np.sum((u3 - u4) ** 2, axis=1)) / c

#     u3_safe = np.clip(u3, 1e-12, 1)
#     u4_safe = np.clip(u4, 1e-12, 1)
#     ce = np.sum(u3_safe * np.log2(u3_safe / u4_safe), axis=1)
#     cross_entropy = np.mean(ce)

#     return E, OA, kappa, mse, cross_entropy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

def FEM(u3, u4, plot_results=False, img_name=None):
    """
    Compares two fuzzy membership matrices with table output.
    
    Parameters:
    - u3, u4: Membership matrices to compare
    - plot_results: If True, generates comparison table/plots
    - img_name: Prefix for saved outputs
    
    Returns: 
    - E: Confusion Matrix
    - OA: Overall Accuracy (%)
    - kappa: Kappa coefficient
    - mse: Mean Squared Error
    - cross_entropy: Cross Entropy
    """
    # Original calculations (unchanged)
    if u3.shape != u4.shape:
        raise ValueError(f"Shape mismatch: {u3.shape} vs {u4.shape}")

    u3 = u3 / np.sum(u3, axis=1, keepdims=True)
    u4 = u4 / np.sum(u4, axis=1, keepdims=True)
    u3 = np.nan_to_num(u3)
    u4 = np.nan_to_num(u4)

    c = u3.shape[1]
    E = np.zeros((c, c))
    for i in range(c):
        for j in range(c):
            E[i, j] = np.sum(np.minimum(u3[:, i], u4[:, j]))

    row_sums = np.sum(E, axis=1)
    col_sums = np.sum(E, axis=0)
    total = np.sum(E)
    diag = np.diag(E)

    OA = 100 * np.sum(diag) / total if total != 0 else 0

    expected_accuracy = np.dot(row_sums, col_sums) / (total ** 2)
    observed_accuracy = np.sum(diag) / total
    kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy) if (1 - expected_accuracy) != 0 else 0

    mse = np.mean(np.sum((u3 - u4) ** 2, axis=1)) / c

    u3_safe = np.clip(u3, 1e-12, 1)
    u4_safe = np.clip(u4, 1e-12, 1)
    ce = np.sum(u3_safe * np.log2(u3_safe / u4_safe), axis=1)
    cross_entropy = np.mean(ce)
    
    # ===== NEW: TABULAR OUTPUT =====
    if plot_results:
        if img_name is None:
            raise ValueError("img_name required when plot_results=True")
            
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)
        ax.axis('off')
        
        # Table data
        metrics = [
            ["Metric", "Value", "Threshold"],
            ["Overall Accuracy", f"{OA:.2f}%", ">85%"],
            ["Kappa Coefficient", f"{kappa:.3f}", ">0.8"],
            ["Mean Squared Error", f"{mse:.4f}", "<0.05"],
            ["Cross Entropy", f"{cross_entropy:.4f}", "<0.1"]
        ]
        
        # Create table
        table = ax.table(cellText=metrics,
                        loc='center',
                        colWidths=[0.3, 0.2, 0.2],
                        cellLoc='center')
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Color cells based on performance
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header
                cell.set_facecolor('#40466e')
                cell.set_text_props(color='white')
            elif j == 1:  # Values
                if i == 1 and OA < 85:
                    cell.set_facecolor('#ffb3ba')  # Red if below threshold
                elif i == 2 and kappa < 0.8:
                    cell.set_facecolor('#ffb3ba')
                elif i == 3 and mse > 0.05:
                    cell.set_facecolor('#ffb3ba')
                elif i == 4 and cross_entropy > 0.1:
                    cell.set_facecolor('#ffb3ba')
                else:
                    cell.set_facecolor('#baffc9')  # Green if meets threshold
        
        plt.title("Fuzzy Membership Comparison Metrics", pad=20)
        plt.savefig(f'results/{img_name}_FEM_table.jpeg', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Additional detailed error table (optional)
        fig = plt.figure(figsize=(12, c*0.5 + 2))
        ax = plt.subplot(111)
        ax.axis('off')
        
        # Per-cluster error data
        cluster_errors = []
        for i in range(c):
            abs_error = np.mean(np.abs(u3[:, i] - u4[:, i]))
            squared_error = np.mean((u3[:, i] - u4[:, i])**2)
            cluster_errors.append([
                f"Cluster {i+1}",
                f"{abs_error:.4f}",
                f"{squared_error:.4f}"
            ])
        
        error_metrics = [
            ["Cluster", "Mean Absolute Error", "Mean Squared Error"]
        ] + cluster_errors
        
        table = ax.table(cellText=error_metrics,
                        loc='center',
                        cellLoc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        plt.title("Per-Cluster Error Analysis", pad=20)
        plt.savefig(f'results/{img_name}_FEM_cluster_errors.jpeg', bbox_inches='tight', dpi=300)
        plt.close()

    return E, OA, kappa, mse, cross_entropy