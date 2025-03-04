import numpy as np
import matplotlib.pyplot as plt
from CIFAR10_helper_functions import *
from datasets import load_dataset

# Load the CIFAR10 training dataset.
train_ds = load_dataset('cifar10', split='train')

# Define the CIFAR10 class names.
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Example usage with the test dataset (assuming test_ds has been loaded):
plot_mnistified_samples(train_ds, cifar10_classes, convert_cifar10_to_mnist_format, filename='figures/mnistified_cifar10.png', save_flag=True)

# Load the prediction NumPy array.
data_np = np.load('scripts/data/cifar10_50k_mnistified_predictions.npy')

# Load centroids
centroids = np.load('/home/daniel/git/work-in-progress/centroids.npy')

# Add the distance to centroid column to data_np
data_np_with_distances = add_distance_to_centroid(data_np, centroids)

# Call the function to plot the distances
plot_distances_to_centroids(data_np_with_distances, cifar10_classes)

# Calculate min, max, and avg distances
min_distances = min_distances_by_predicted_class(data_np_with_distances)
print(min_distances)

max_distances = max_distances_by_predicted_class(data_np_with_distances)
print(max_distances)

avg_distances = avg_distances_by_predicted_class(data_np_with_distances)
print(avg_distances)

# Plot min, max, and avg distances
plot_min_max_avg_distances(min_distances, max_distances, avg_distances, cifar10_classes, 
                          filename="figures/mnistified_cifar10_min_max_avg_distances_to_centroids.png", save=True)

# Plot heatmaps
plot_heatmap(data_np_with_distances, cifar10_classes, filename="figures/mnistified_cifar10_heatmap.png", save=True)
plot_heatmap_percentages(data_np_with_distances, cifar10_classes, filename="figures/mnistified_cifar10_heatmap_percentages.png", save=True)

# Create and display threshold table
alpha_threshold_table = create_cifar_mnist_table(data_np_with_distances)

# Display the results
print("CIFAR 10 MNIST Prediction Threshold Table:")
print("-" * 120)
header = ["Threshold"] + cifar10_classes + ["TBT", "TAT"]
print("{:<10} ".format(header[0]), end="")
for i in range(1, len(header)):
    print("{:<10} ".format(header[i]), end="")
print()
print("-" * 120)

for row in alpha_threshold_table:
    print("{:<10.2f} ".format(row[0]), end="")
    for i in range(1, len(row)):
        print("{:<10.0f} ".format(row[i]), end="")
    print()

# Save table as numpy array
np.save('data/copilot_cifar_10_mnist_prediction_thresholds.npy', alpha_threshold_table)

# Plot average softmax values
plot_cifar10_mnist_prediction_avg_softmax(data_np_with_distances, cifar10_classes, 
                                         filename="figures/cifar10_mnist_prediction_avg_softmax.png", save=True)

# TODO
# 1. Add function calls to latex file referencing tables and plots
# 2. Sanity check 5 random totals


