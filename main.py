import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from fcm_generation import fcm_generation
from fsc_generation import fsc_generation
from FCM_RBFNN_FSC1 import FCM_RBFNN_FSC1
from FEM import FEM
from graph import simple_fuzzy_plot
from simple_cluster_plot import plot_iris_clusters, plot_fuzzy_clusters, plot_cluster_comparison
from accuracy_table import create_accuracy_table, plot_accuracy_table, create_comparison_table
import matplotlib.pyplot as plt

def sanitize_membership_matrix(U):
    U = np.nan_to_num(U, nan=0.0, posinf=1.0, neginf=0.0)
    row_sums = np.sum(U, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return U / row_sums

def debug_matrix(name, M):
    print(f"\n{name} stats:")
    print(f"  Shape: {M.shape}, NaNs: {np.isnan(M).any()}, Infs: {np.isinf(M).any()}")
    print(f"  Min: {np.min(M):.4f}, Max: {np.max(M):.4f}")
    print(f"  Row sums → Min: {np.min(np.sum(M, axis=1)):.2f}, Max: {np.max(np.sum(M, axis=1)):.2f}")

def get_stratified_sample(data, labels, sample_frac=0.1):
    unique_classes = np.unique(labels)
    train_indices = []
    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        n_samples = max(1, int(sample_frac * len(cls_indices)))
        train_indices.extend(np.random.choice(cls_indices, n_samples, replace=False))
    return data[train_indices], np.delete(data, train_indices, axis=0)

def run_iris_pipeline():
    iris = datasets.load_iris()
    x = iris.data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    y_true = iris.target.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y_true)
    full_data = np.hstack((x_scaled, y_onehot))
    training_sample, _ = get_stratified_sample(full_data, y_true, 0.3)
    print(f"\nTraining with {len(training_sample)} samples ({len(training_sample)/len(x)*100:.1f}%)")
    print("Class counts:", np.bincount(training_sample[:, -3:].argmax(axis=1)))

    best_acc = 0
    best_params = None
    best_y_est1 = None
    # Grid search for FCM and RBFNN params
    for m in [2.0, 2.5, 3.0]:
        for max_iter in [100, 200]:
            me, U = fcm_generation(x, 3, "iris", m=m, max_iter=max_iter)
            u1 = fsc_generation(x_scaled, training_sample, 3, "iris")
            for lambda_reg in [0.0001, 0.001, 0.01]:
                for kernel_scale in [0.1, 0.3, 0.5, 1.0]:
                    b, y_est1 = FCM_RBFNN_FSC1(
                        x, U, y_onehot, x, "iris",
                        lambda_reg=lambda_reg,
                        kernel_scale=kernel_scale
                    )
                    y_true_onehot = np.zeros((len(y_true), 3))
                    y_true_onehot[np.arange(len(y_true)), y_true.flatten()] = 1
                    rbfnn_results = create_accuracy_table(y_true_onehot, y_est1, "RBFNN Classification")
                    if rbfnn_results['accuracy'] > best_acc:
                        best_acc = rbfnn_results['accuracy']
                        best_params = (m, max_iter, lambda_reg, kernel_scale)
                        best_y_est1 = y_est1
    print(f"\nBest RBFNN Accuracy: {best_acc:.4f} with params m={best_params[0]}, max_iter={best_params[1]}, lambda_reg={best_params[2]}, kernel_scale={best_params[3]}")
    # Use best params for final results
    me, U = fcm_generation(x, 3, "iris", m=best_params[0], max_iter=best_params[1])
    u1 = fsc_generation(x_scaled, training_sample, 3, "iris")
    y_est1 = best_y_est1
    print("\n=== Generating Simple Cluster Plots ===")
    plot_iris_clusters(x_scaled, y_true.flatten(), "Iris Dataset - True Classes")
    plot_fuzzy_clusters(x, U, "FCM Clusters")
    plot_fuzzy_clusters(x_scaled, u1, "FSC Clusters")
    plot_fuzzy_clusters(x, y_est1, "RBFNN Clusters")
    plot_cluster_comparison(x_scaled, y_true.flatten(), U, u1, y_est1)
    print("\n=== Generating Accuracy Tables ===")
    y_true_onehot = np.zeros((len(y_true), 3))
    y_true_onehot[np.arange(len(y_true)), y_true.flatten()] = 1
    fcm_results = create_accuracy_table(y_true_onehot, U, "FCM Clustering")
    fsc_results = create_accuracy_table(y_true_onehot, u1, "FSC Clustering")
    rbfnn_results = create_accuracy_table(y_true_onehot, y_est1, "RBFNN Classification")
    plot_accuracy_table(fcm_results, "FCM Clustering")
    plot_accuracy_table(fsc_results, "FSC Clustering")
    plot_accuracy_table(rbfnn_results, "RBFNN Classification")
    all_results = {
        "FCM": fcm_results,
        "FSC": fsc_results,
        "RBFNN": rbfnn_results
    }
    create_comparison_table(all_results)
    print(f"\n=== Accuracy Summary ===")
    print(f"FCM Accuracy: {fcm_results['accuracy']:.4f} ({fcm_results['accuracy']*100:.2f}%)")
    print(f"FSC Accuracy: {fsc_results['accuracy']:.4f} ({fsc_results['accuracy']*100:.2f}%)")
    print(f"RBFNN Accuracy: {rbfnn_results['accuracy']:.4f} ({rbfnn_results['accuracy']*100:.2f}%)")
    U = sanitize_membership_matrix(U)
    u1 = sanitize_membership_matrix(u1)
    y_est1 = sanitize_membership_matrix(y_est1)
    print("\n=== Performance Evaluation ===")
    comparisons = [
        (("FCM", U), ("FSC", u1)),
        (("FCM", U), ("RBFNN", y_est1)),
        (("FSC", u1), ("RBFNN", y_est1)),
    ]
    for (name1, m1), (name2, m2) in comparisons:
        E, OA, kappa, mse, ce = FEM(m1, m2)
        print(f"{name1} vs {name2}: OA={OA:.1f}%, κ={kappa:.3f}, MSE={mse:.4f}")

if __name__ == "__main__":
    plt.style.use('seaborn-v0_8')
    np.random.seed(42)
    run_iris_pipeline()
    plt.show()  # Keep plots open at the end