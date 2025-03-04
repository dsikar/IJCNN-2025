import numpy as np
import matplotlib.pyplot as plt
from CIFAR10_helper_functions import *
# from helper_functions import convert_cifar10_to_mnist_format
from datasets import load_dataset

# Load the CIFAR10 training dataset.
train_ds = load_dataset('cifar10', split='train')

#########################################
# PLOT ROW OF MNISTified CIFAR10 IMAGES #
#########################################

#     output_path='cifar10_predictions.csv',
#     device='cpu'  # or 'cuda'
# )

# data_np = results['data_np']
# df = results['dataframe']

# # Save the prediction NumPy array for further analysis.
# np.save('data/cifar10_50k_mnistified_predictions.npy', data_np)


# Define the CIFAR10 class names.
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Example usage with the test dataset (assuming test_ds has been loaded):
plot_mnistified_samples(train_ds, cifar10_classes, convert_cifar10_to_mnist_format, filename='figures/mnistified_cifar10.png', save_flag=True)

# Load the prediction NumPy array.
data_np = np.load('scripts/data/cifar10_50k_mnistified_predictions.npy')

# Load CIFAR10 labels
# cifar10_labels = np.load('data/cifar10_predictions.csv')
# Load centroids
centroids = np.load('/home/daniel/git/work-in-progress/centroids.npy')

# Add the distance to centroid column to data_np

data_np_with_distances = add_distance_to_centroid(data_np, centroids)
# 3 plots required
# 1. Triple bar plot per centroid, nearest, average and furthest.
# 2. Softmax averages
# 3. Heatmap classes x digits

# 1. Triple bar plot per centroid, nearest, average and furthest.
plot_distances_to_centroids(data_np_with_distances, cifar10_classes)

# Example usage
min_distances = min_distances_by_predicted_class(data_np_with_distances)
print(min_distances)

# Example usage
max_distances = max_distances_by_predicted_class(data_np_with_distances)
print(max_distances)

# Example usage
avg_distances = avg_distances_by_predicted_class(data_np_with_distances)
print(avg_distances)

# Example usage
plot_min_max_avg_distances(min_distances, max_distances, avg_distances, cifar10_classes, filename="figures/mnistified_cifar10_min_max_avg_distances_to_centroids.png", save=True)

# Example usage
plot_heatmap(data_np_with_distances, cifar10_classes, filename="figures/mnistified_cifar10_heatmap.png", save=True)
# Total counts in confusion matrix: 50000.0
# Heatmap saved as figures/mnistified_cifar10_heatmap.png

plot_heatmap_percentages(data_np_with_distances, cifar10_classes, filename="figures/mnistified_cifar10_heatmap_percentages.png", save=True)
# Heatmap saved as figures/mnistified_cifar10_heatmap_percentages.png

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

# CIFAR 10 MNIST Prediction Threshold Table:
# ------------------------------------------------------------------------------------------------------------------------
# Threshold  airplane   automobile bird       cat        deer       dog        frog       horse      ship       truck      TBT        TAT        
# ------------------------------------------------------------------------------------------------------------------------
# 0.90       717        715        4244       6716       3331       6415       753        3364       23411      334        50000      0          
# 0.80       673        668        4125       6605       3231       6258       691        3231       23309      319        49110      890        
# 0.70       489        485        3279       5548       2663       5256       491        2531       21799      233        42774      7226       
# 0.60       292        330        2274       4068       1891       4031       309        1756       18611      135        33697      16303      
# 0.50       190        216        1564       2935       1348       3106       192        1212       15464      91         26318      23682      
# 0.40       108        137        1032       1965       916        2296       115        785        12142      55         19551      30449      
# 0.30       50         78         643        1228       592        1589       54         504        8845       34         13617      36383      
# 0.20       22         42         364        662        361        1042       32         263        5428       18         8234       41766      
# 0.10       4          7          154        232        149        511        12         91         2270       2          3432       46568      
# 0.05       1          1          60         96         70         278        3          33         970        1          1513       48487      
# 0.02       0          0          31         32         22         131        0          11         280        0          507        49493     


plot_cifar10_mnist_prediction_avg_softmax(data_np_with_distances, cifar10_classes, filename="figures/cifar10_mnist_prediction_avg_softmax.png", save=True)
# Saved average softmax outputs for CIFAR10 predictions. figures/cifar10_mnist_prediction_avg_softmax.png

# DONE
# 1. Move all function definitions to CIFAR10_helper_functions.py
# TODO
# 2. Add function calls to latex file referencing tables and plots
# 3. sanity check 5 random totals
