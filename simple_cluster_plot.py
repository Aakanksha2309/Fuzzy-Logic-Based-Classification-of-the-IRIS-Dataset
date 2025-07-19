import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def plot_iris_clusters(x, y_true, title="Iris Dataset - True Classes"):
    """Simple scatter plot showing the 3 Iris classes"""
    iris = datasets.load_iris()
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Create subplots for different feature combinations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    # Feature combinations to plot
    feature_pairs = [
        (0, 1, "Sepal Length vs Sepal Width"),
        (0, 2, "Sepal Length vs Petal Length"), 
        (0, 3, "Sepal Length vs Petal Width"),
        (2, 3, "Petal Length vs Petal Width")
    ]
    
    colors = ['red', 'blue', 'green']
    
    for idx, (f1, f2, label) in enumerate(feature_pairs):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        # Plot each class with different color
        for i in range(3):
            mask = y_true == i
            ax.scatter(x[mask, f1], x[mask, f2], 
                      c=colors[i], label=target_names[i], alpha=0.7)
        
        ax.set_xlabel(feature_names[f1])
        ax.set_ylabel(feature_names[f2])
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/iris_true_clusters.jpeg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_fuzzy_clusters(x, membership_matrix, title="Fuzzy Clusters"):
    """Plot fuzzy membership as colored points"""
    iris = datasets.load_iris()
    feature_names = iris.feature_names
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    feature_pairs = [
        (0, 1, "Sepal Length vs Sepal Width"),
        (0, 2, "Sepal Length vs Petal Length"), 
        (0, 3, "Sepal Length vs Petal Width"),
        (2, 3, "Petal Length vs Petal Width")
    ]
    
    colors = ['red', 'blue', 'green']
    
    for idx, (f1, f2, label) in enumerate(feature_pairs):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        # Get dominant cluster for each point
        cluster_labels = np.argmax(membership_matrix, axis=1)
        
        # Plot with cluster colors
        for i in range(3):
            mask = cluster_labels == i
            ax.scatter(x[mask, f1], x[mask, f2], 
                      c=colors[i], label=f'Cluster {i+1}', alpha=0.7)
        
        ax.set_xlabel(feature_names[f1])
        ax.set_ylabel(feature_names[f2])
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{title.lower().replace(" ", "_")}.jpeg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_cluster_comparison(x, y_true, U_fcm, U_fsc, U_rbfnn):
    """Compare all three clustering methods"""
    iris = datasets.load_iris()
    target_names = iris.target_names
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Cluster Comparison: True vs FCM vs FSC vs RBFNN", fontsize=16)
    
    # Use Petal Length vs Petal Width for comparison (most separable)
    f1, f2 = 2, 3  # Petal Length vs Petal Width
    
    colors = ['red', 'blue', 'green']
    
    # True classes
    ax = axes[0, 0]
    for i in range(3):
        mask = y_true == i
        ax.scatter(x[mask, f1], x[mask, f2], c=colors[i], label=target_names[i], alpha=0.7)
    ax.set_title("True Classes")
    ax.set_xlabel("Petal Length")
    ax.set_ylabel("Petal Width")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # FCM clusters
    ax = axes[0, 1]
    cluster_labels = np.argmax(U_fcm, axis=1)
    for i in range(3):
        mask = cluster_labels == i
        ax.scatter(x[mask, f1], x[mask, f2], c=colors[i], label=f'FCM Cluster {i+1}', alpha=0.7)
    ax.set_title("FCM Clusters")
    ax.set_xlabel("Petal Length")
    ax.set_ylabel("Petal Width")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # FSC clusters
    ax = axes[0, 2]
    cluster_labels = np.argmax(U_fsc, axis=1)
    for i in range(3):
        mask = cluster_labels == i
        ax.scatter(x[mask, f1], x[mask, f2], c=colors[i], label=f'FSC Cluster {i+1}', alpha=0.7)
    ax.set_title("FSC Clusters")
    ax.set_xlabel("Petal Length")
    ax.set_ylabel("Petal Width")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RBFNN clusters
    ax = axes[1, 0]
    cluster_labels = np.argmax(U_rbfnn, axis=1)
    for i in range(3):
        mask = cluster_labels == i
        ax.scatter(x[mask, f1], x[mask, f2], c=colors[i], label=f'RBFNN Cluster {i+1}', alpha=0.7)
    ax.set_title("RBFNN Clusters")
    ax.set_xlabel("Petal Length")
    ax.set_ylabel("Petal Width")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Hide remaining subplots
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/cluster_comparison.jpeg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close() 