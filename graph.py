import numpy as np
import matplotlib.pyplot as plt

def simple_fuzzy_plot(data, plot_type, img_name, **kwargs):
    """
    Unified simple plotting for all fuzzy functions.
    
    Parameters:
    - data: Input array (membership matrix/cluster assignments)
    - plot_type: 'membership', 'cluster_map', or 'radial'
    - img_name: Output filename prefix
    - kwargs: Additional options (cluster_centers, clims, etc.)
    """
    plt.figure(figsize=(8, 6))
    
    if plot_type == 'membership':
        # Basic membership value plot
        for i in range(data.shape[1]):
            plt.plot(data[:, i], alpha=0.7, label=f'Cluster {i+1}')
        plt.title('Membership Values')
        plt.xlabel('Samples')
        plt.ylabel('Membership')
        plt.legend()
        
    elif plot_type == 'cluster_map':
        # Cluster assignment plot (like your current imshow)
        clims = kwargs.get('clims', [1, data.max()])
        plt.imshow(data, cmap='jet', vmin=clims[0], vmax=clims[1])
        plt.colorbar()
        if kwargs.get('is_image', True):
            plt.axis('off')
        plt.title('Cluster Assignments')
        
    elif plot_type == 'radial':
        # Simplified radial plot
        theta = np.linspace(0, 2*np.pi, data.shape[1], endpoint=False)
        for i in range(min(20, data.shape[0])):  # Only plot first 20 samples
            plt.polar(theta, data[i], 'o-', alpha=0.3)
        plt.title('Radial Membership View')
    
    plt.tight_layout()
    plt.savefig(f'results/{img_name}.jpeg', bbox_inches='tight')
    plt.show()  # Show plot on screen
    plt.close()