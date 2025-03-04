import numpy as np
import matplotlib.pyplot as plt
from CIFAR10_helper_functions import *
from datasets import load_dataset

# Load the CIFAR10 training dataset
train_ds = load_dataset('cifar10', split='train')

# Define CIFAR10 class names
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Plot MNISTified samples
plot_mnistified_samples(train_ds, cifar10_classes, convert_cifar10_to_mnist_format, 
                       filename='figures/mnistified_cifar10.png', save_flag=True)

# Load prediction data and centroids
data_np = np.load('scripts/data/cifar10_50k_mnistified_predictions.npy')
centroids = np.load('/home/daniel/git/work-in-progress/centroids.npy')

# Add distance to centroid column
data_np_with_distances = add_distance_to_centroid(data_np, centroids)
# Generate and save plots
def plot_distances_to_centroids(data_np_with_distances, cifar10_classes):
    """
    Plot the distances to centroids for CIFAR10 images.
    
    Parameters:
    -----------
    data_np_with_distances : numpy.ndarray
        Array containing distances to centroids.
    cifar10_classes : list
        List of CIFAR10 class names.
    """
    # Create figure and axis
    plt.figure(figsize=(15, 7))
    
    # Set the positions for the bars
    x = np.arange(len(cifar10_classes))
    width = 0.25
    
    # Extract distances
    nearest = data_np_with_distances[:, 0][:len(cifar10_classes)]
    average = data_np_with_distances[:, 1][:len(cifar10_classes)]
    furthest = data_np_with_distances[:, 2][:len(cifar10_classes)]
    
    # Create the bars
    plt.bar(x - width, nearest, width, label='Near', color='skyblue')
    plt.bar(x, average, width, label='Avg', color='lightgreen')
    plt.bar(x + width, furthest, width, label='Far', color='lightcoral')
    
    # Customize the plot
    plt.xlabel('Centroid (Class)')
    plt.ylabel('Distance to Centroid')
    plt.title('Distances to Centroids for CIFAR10 Images')
    plt.legend()
    
    # Set x-ticks with class names and indices
    plt.xticks(x, [f'{cifar10_classes[i]} ({i})' for i in range(len(cifar10_classes))], rotation=45)
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis to logarithmic scale
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

# Calculate distances statistics
min_distances = min_distances_by_predicted_class(data_np_with_distances)
max_distances = max_distances_by_predicted_class(data_np_with_distances)
avg_distances = avg_distances_by_predicted_class(data_np_with_distances)

# Generate and save plots
plot_min_max_avg_distances(min_distances, max_distances, avg_distances, cifar10_classes,
                          filename="figures/mnistified_cifar10_min_max_avg_distances_to_centroids.png", save=True)

plot_heatmap(data_np_with_distances, cifar10_classes, 
            filename="figures/mnistified_cifar10_heatmap.png", save=True)

plot_heatmap_percentages(data_np_with_distances, cifar10_classes,
                        filename="figures/mnistified_cifar10_heatmap_percentages.png", save=True)

# Create and save threshold table
alpha_threshold_table = create_cifar_mnist_table(data_np_with_distances)
np.save('data/copilot_cifar_10_mnist_prediction_thresholds.npy', alpha_threshold_table)

# Plot average softmax values
plot_cifar10_mnist_prediction_avg_softmax(data_np_with_distances, cifar10_classes,
                                         filename="figures/cifar10_mnist_prediction_avg_softmax.png", save=True)



