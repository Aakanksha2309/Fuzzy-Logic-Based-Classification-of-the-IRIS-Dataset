import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def create_accuracy_table(y_true, y_pred, method_name="Method"):
    """Create detailed accuracy table with true vs predicted labels"""
    
    # Convert to class labels
    if len(y_true.shape) > 1:
        y_true_labels = np.argmax(y_true, axis=1)
    else:
        y_true_labels = y_true
        
    if len(y_pred.shape) > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_pred_labels = y_pred
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    error_rate = 1 - accuracy
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    
    # Get classification report
    report = classification_report(y_true_labels, y_pred_labels, 
                                 target_names=['Setosa', 'Versicolor', 'Virginica'],
                                 output_dict=True)
    
    return {
        'accuracy': accuracy,
        'error_rate': error_rate,
        'confusion_matrix': cm,
        'classification_report': report,
        'true_labels': y_true_labels,
        'predicted_labels': y_pred_labels
    }

def plot_accuracy_table(results_dict, method_name="Method"):
    """Plot accuracy table with detailed metrics"""
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{method_name} - Accuracy Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall Metrics Table
    ax1 = axes[0, 0]
    ax1.axis('off')
    
    metrics_data = [
        ['Metric', 'Value', 'Percentage'],
        ['Accuracy', f'{results_dict["accuracy"]:.4f}', f'{results_dict["accuracy"]*100:.2f}%'],
        ['Error Rate', f'{results_dict["error_rate"]:.4f}', f'{results_dict["error_rate"]*100:.2f}%'],
        ['Correct Predictions', f'{np.sum(results_dict["true_labels"] == results_dict["predicted_labels"])}', ''],
        ['Total Samples', f'{len(results_dict["true_labels"])}', '']
    ]
    
    table1 = ax1.table(cellText=metrics_data, loc='center', cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(12)
    table1.scale(1, 2)
    
    # Color the header
    for i in range(3):
        table1[(0, i)].set_facecolor('#40466e')
        table1[(0, i)].set_text_props(color='white', weight='bold')
    
    # Color accuracy row
    table1[(1, 0)].set_facecolor('#baffc9')
    table1[(1, 1)].set_facecolor('#baffc9')
    table1[(1, 2)].set_facecolor('#baffc9')
    
    ax1.set_title('Overall Performance Metrics', pad=20, fontweight='bold')
    
    # 2. Confusion Matrix
    ax2 = axes[0, 1]
    cm = results_dict['confusion_matrix']
    
    # Create heatmap using matplotlib
    im = ax2.imshow(cm, cmap='Blues', interpolation='nearest')
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, str(cm[i, j]), ha='center', va='center', 
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontweight='bold')
    
    # Set labels
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(['Setosa', 'Versicolor', 'Virginica'])
    ax2.set_yticklabels(['Setosa', 'Versicolor', 'Virginica'])
    ax2.set_title('Confusion Matrix', fontweight='bold')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    # Add colorbar
    plt.colorbar(im, ax=ax2)
    
    # 3. Per-Class Metrics
    ax3 = axes[1, 0]
    ax3.axis('off')
    
    report = results_dict['classification_report']
    class_metrics = []
    for class_name in ['Setosa', 'Versicolor', 'Virginica']:
        if class_name in report:
            class_data = report[class_name]
            class_metrics.append([
                class_name,
                f'{class_data["precision"]:.3f}',
                f'{class_data["recall"]:.3f}',
                f'{class_data["f1-score"]:.3f}',
                f'{class_data["support"]}'
            ])
    
    class_table_data = [
        ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    ] + class_metrics
    
    table3 = ax3.table(cellText=class_table_data, loc='center', cellLoc='center')
    table3.auto_set_font_size(False)
    table3.set_fontsize(11)
    table3.scale(1, 1.5)
    
    # Color header
    for i in range(5):
        table3[(0, i)].set_facecolor('#40466e')
        table3[(0, i)].set_text_props(color='white', weight='bold')
    
    ax3.set_title('Per-Class Performance', pad=20, fontweight='bold')
    
    # 4. True vs Predicted Comparison
    ax4 = axes[1, 1]
    true_labels = results_dict['true_labels']
    pred_labels = results_dict['predicted_labels']
    
    # Create comparison plot
    x = np.arange(len(true_labels))
    correct_mask = true_labels == pred_labels
    
    # Plot correct predictions in green, incorrect in red
    ax4.scatter(x[correct_mask], true_labels[correct_mask], 
               c='green', alpha=0.7, label='Correct', s=30)
    ax4.scatter(x[~correct_mask], true_labels[~correct_mask], 
               c='red', alpha=0.7, label='Incorrect', s=30)
    
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Class Label')
    ax4.set_title('True vs Predicted Labels')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['Setosa', 'Versicolor', 'Virginica'])
    
    plt.tight_layout()
    plt.savefig(f'results/{method_name.lower().replace(" ", "_")}_accuracy_table.jpeg', 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def create_comparison_table(all_results):
    """Create comparison table for all methods"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.axis('off')
    
    # Prepare comparison data
    comparison_data = []
    for method_name, results in all_results.items():
        comparison_data.append([
            method_name,
            f'{results["accuracy"]:.4f}',
            f'{results["accuracy"]*100:.2f}%',
            f'{results["error_rate"]:.4f}',
            f'{results["error_rate"]*100:.2f}%',
            f'{np.sum(results["true_labels"] == results["predicted_labels"])}',
            f'{len(results["true_labels"])}'
        ])
    
    table_data = [
        ['Method', 'Accuracy', 'Accuracy %', 'Error Rate', 'Error %', 'Correct', 'Total']
    ] + comparison_data
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Color header
    for i in range(7):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(color='white', weight='bold')
    
    # Color best performing method
    accuracies = [results["accuracy"] for results in all_results.values()]
    best_idx = np.argmax(accuracies)
    for i in range(7):
        table[(best_idx + 1, i)].set_facecolor('#baffc9')
    
    ax.set_title('Method Comparison - Accuracy Analysis', pad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/method_comparison_table.jpeg', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close() 