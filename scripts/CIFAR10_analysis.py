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

# Call the function to plot the distances
plot_distances_to_centroids(data_np_with_distances, cifar10_classes)

def min_distances_by_predicted_class(data_np_with_distances):
    """
    Calculate the minimum distances to prediction centroids grouped by predicted class.
    
    Parameters:
    -----------
    data_np_with_distances : numpy.ndarray
        Array containing distances to centroids and predicted classes.
    
    Returns:
    --------
    min_distances : numpy.ndarray
        Array containing the minimum distances for each predicted class.
    """
    min_distances = np.zeros(10)
    
    for i in range(10):
        class_distances = data_np_with_distances[(data_np_with_distances[:, 10] == j) & (data_np_with_distances[:, 11] == i), 12]
        if len(class_distances) > 0:
            min_distances[i] = np.min(class_distances)
        else:
            min_distances[i] = np.nan  # Handle case where no distances are found for a class
    
    return min_distances

# Example usage
min_distances = min_distances_by_predicted_class(data_np_with_distances)
print(min_distances)

def max_distances_by_predicted_class(data_np_with_distances):
    """
    Calculate the maximum distances to prediction centroids grouped by predicted class.
    
    Parameters:
    -----------
    data_np_with_distances : numpy.ndarray
        Array containing distances to centroids and predicted classes.
    
    Returns:
    --------
    max_distances : numpy.ndarray
        Array containing the maximum distances for each predicted class.
    """
    max_distances = np.zeros(10)
    
    for i in range(10):
        class_distances = data_np_with_distances[data_np_with_distances[:, 11] == i, 12]
        if len(class_distances) > 0:
            max_distances[i] = np.max(class_distances)
        else:
            max_distances[i] = np.nan  # Handle case where no distances are found for a class
    
    return max_distances

# Example usage
max_distances = max_distances_by_predicted_class(data_np_with_distances)
print(max_distances)

def avg_distances_by_predicted_class(data_np_with_distances):
    """
    Calculate the average distances to prediction centroids grouped by predicted class.
    
    Parameters:
    -----------
    data_np_with_distances : numpy.ndarray
        Array containing distances to centroids and predicted classes.
    
    Returns:
    --------
    avg_distances : numpy.ndarray
        Array containing the average distances for each predicted class.
    """
    avg_distances = np.zeros(10)
    
    for i in range(10):
        class_distances = data_np_with_distances[data_np_with_distances[:, 11] == i, 12]
        if len(class_distances) > 0:
            avg_distances[i] = np.mean(class_distances)
        else:
            avg_distances[i] = np.nan  # Handle case where no distances are found for a class
    
    return avg_distances

# Example usage
avg_distances = avg_distances_by_predicted_class(data_np_with_distances)
print(avg_distances)

def plot_min_max_avg_distances(min_distances, max_distances, avg_distances, cifar10_classes, filename=None, save=False):
    """
    Plot the minimum, maximum, and average distances to centroids for CIFAR10 images.
    
    Parameters:
    -----------
    min_distances : numpy.ndarray
        Array containing the minimum distances for each predicted class.
    max_distances : numpy.ndarray
        Array containing the maximum distances for each predicted class.
    avg_distances : numpy.ndarray
        Array containing the average distances for each predicted class.
    cifar10_classes : list
        List of CIFAR10 class names.
    filename : str, optional
        The filename to save the plot. Default is None.
    save : bool, optional
        Flag to save the plot. Default is False.
    """
    # Create figure and axis
    plt.figure(figsize=(15, 7))
    
    # Set the positions for the bars
    x = np.arange(len(cifar10_classes))
    width = 0.25
    
    # Create the bars
    plt.bar(x - width, min_distances, width, label='Min', color='skyblue')
    plt.bar(x, avg_distances, width, label='Avg', color='lightgreen')
    plt.bar(x + width, max_distances, width, label='Max', color='lightcoral')
    
    # Customize the plot
    plt.xlabel('Predicted Class')
    plt.ylabel('Distance to Centroid')
    plt.title('Min, Max, and Avg Distances to MNIST Centroids for MNISTified CIFAR10 Images')
    plt.legend()
    
    # Set x-ticks with class names and indices
    plt.xticks(x, [f'{cifar10_classes[i]} ({i})' for i in range(len(cifar10_classes))], rotation=45)
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis to logarithmic scale
    plt.yscale('log')
    
    plt.tight_layout()
    
    # Save the plot if required
    if save and filename:
        plt.savefig(filename)
        print(f"Plot saved as {filename}")
    else:
        plt.show()

# Example usage
plot_min_max_avg_distances(min_distances, max_distances, avg_distances, cifar10_classes, filename="figures/mnistified_cifar10_min_max_avg_distances_to_centroids.png", save=True)

def plot_heatmap(data_np_with_distances, cifar10_classes, filename=None, save=False):
    """
    Plot a heatmap of the confusion matrix for CIFAR10 images.
    
    Parameters:
    -----------
    data_np_with_distances : numpy.ndarray
        Array containing the data where:
        - Columns 0-9: Softmax outputs
        - Column 10: True class
        - Column 11: Predicted class
        - Column 12: Distance to true class centroid        
    cifar10_classes : list
        List of CIFAR10 class names.
    filename : str, optional
        The filename to save the plot. Default is None.
    save : bool, optional
        Flag to save the plot. Default is False.
    """
    # Initialize the confusion matrix
    confusion_matrix = np.zeros((len(cifar10_classes), len(cifar10_classes)))
    # i: Index for the true class
    # j: Index for the predicted class
    for i in range(len(cifar10_classes)):
        for j in range(len(cifar10_classes)):
            confusion_matrix[i, j] = np.sum((data_np_with_distances[:, 10] == i) & (data_np_with_distances[:, 11] == j))
    
    # Sanity check print to count the totals in the confusion matrix
    print(f"Total counts in confusion matrix: {np.sum(confusion_matrix)}")
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(confusion_matrix, cmap='Reds', aspect='auto')
    
    # Customize the plot
    plt.colorbar(label='Number of Classifications')
    plt.xlabel('MNIST Predicted Class')
    plt.ylabel('CIFAR10 True Class')
    plt.title('Confusion Matrix for CIFAR10 Images')
    plt.xticks(np.arange(len(cifar10_classes)), range(10), rotation=0)
    plt.yticks(np.arange(len(cifar10_classes)), cifar10_classes)
    
    # Annotate the heatmap
    for i in range(len(cifar10_classes)):
        for j in range(len(cifar10_classes)):
            plt.text(j, i, int(confusion_matrix[i, j]), ha='center', va='center', color='black')
    
    plt.tight_layout()
    
    # Save the plot if required
    if save and filename:
        plt.savefig(filename)
        print(f"Heatmap saved as {filename}")
    else:
        plt.show()

# Example usage
plot_heatmap(data_np_with_distances, cifar10_classes, filename="figures/mnistified_cifar10_heatmap.png", save=True)
# Total counts in confusion matrix: 50000.0
# Heatmap saved as figures/mnistified_cifar10_heatmap.png

def create_cifar_mnist_table(data_np):
    """
    Creates a summary table for CIFAR10 classes from the provided data_np array.

    The input data_np is assumed to have the following columns:
      - Columns 0-9: Softmax outputs (for digits 0-9)
      - Column 10: True class (0-9 for digits, >=10 for alphabetic characters)
      - Column 11: Predicted class (always a digit 0-9)
      - Column 12: Distance to true class centroid
      - Column 13: Distance to predicted class centroid

    The output table has 13 columns:
      - Column 0: "Threshold" with values [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02]
      - Columns 1-10: Counts for each predicted CIFAR10 class where the distance (column 13) is <= threshold.
      - Column 11: "TBT" (Total Below Threshold) = total rows with distance <= threshold.
      - Column 12: "TAT" (Total Above Threshold) = total rows with distance > threshold.
    """
    # Define the thresholds (in order: 0.8, 0.7, ..., 0.1, then 0.05, 0.02)
    thresholds = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02])
    
    # Filter to only include rows for CIFAR10 classes.
    alpha_mask = data_np[:, 10] <= 9
    alpha_data = data_np[alpha_mask]
    
    # For CIFAR10 rows, use the predicted class (column 11) and
    # the distance to the predicted class centroid (column 13).
    predicted_classes = alpha_data[:, 11].astype(int)  # Ensure integer type for class matching.
    distances = alpha_data[:, 12]
    
    table_rows = []
    
    # Loop over each threshold value.
    for thresh in thresholds:
        row = [thresh]
        # For each predicted class (0 to 9), count rows with distance <= thresh.
        class_counts = [
            np.sum((predicted_classes == d) & (distances <= thresh))
            for d in range(10)
        ]
        row.extend(class_counts)
        # TBT: Total Below Threshold (all rows with distance <= thresh).
        tbt = np.sum(distances <= thresh)
        # TAT: Total Above Threshold (all rows with distance > thresh).
        tat = np.sum(distances > thresh)
        row.extend([tbt, tat])
        table_rows.append(row)
    
    # Convert to numpy array and return.
    return np.array(table_rows)

# Example usage:
#table = create_alphabetic_table(data_np)
# print(table)

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


def plot_cifar10_mnist_prediction_avg_softmax(data_np_with_distances, cifar10_classes, filename=None, save=False):
    """
    Plot the average softmax values for CIFAR10 images grouped by predicted class.
    
    Parameters:
    -----------
    data_np_with_distances : numpy.ndarray
        Array containing the data where:
        - Columns 0-9: Softmax outputs
        - Column 10: True class
        - Column 11: Predicted class
        - Column 12: Distance to true class centroid        
    cifar10_classes : list
        List of CIFAR10 class names.
    filename : str, optional
        The filename to save the plot. Default is None.
    save : bool, optional
        Flag to save the plot. Default is False.
    """
    # Get the unique predicted classes (digits 0-9)
    labels = np.unique(data_np_with_distances[:, 11]).astype(int)

    # Create a figure and subplots for each predicted class
    fig, axs = plt.subplots(1, len(labels), figsize=(20, 5))
    fig.suptitle('Softmax Average Distributions for MNISTified CIFAR10 Predictions', fontsize=16)

    # Plot average softmax values for each predicted class
    for i, label in enumerate(labels):
        # Get the predictions for the current class
        class_predictions = data_np_with_distances[data_np_with_distances[:, 11] == label, :10]
        # Calculate the average value for each index
        averages = np.mean(class_predictions, axis=0)
        # Plot the bar graph for the current class
        axs[i].bar(np.arange(10), averages, color='lightcoral')
        # Set the y-axis to logarithmic scale
        axs[i].set_yscale('log')
        # Set the y-axis limits to start from 10^-4
        axs[i].set_ylim(bottom=1e-4)
        # Set the title for the current subplot
        axs[i].set_title(f'{cifar10_classes[label].capitalize()} ({label})', fontsize=12)
        # Set the x-tick positions and labels
        axs[i].set_xticks(np.arange(10))
        axs[i].set_xticklabels(np.arange(10), fontsize=10)
        # Add x-axis grid lines
        axs[i].set_xticks(np.arange(10), minor=True)
        axs[i].grid(True, which='minor', axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
        # Add y-axis grid lines
        axs[i].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # Set x-axis label at the bottom of the figure
    fig.text(0.5, 0.04, 'Class Index', ha='center', fontsize=14)

    # Set y-axis label on the left side of the figure
    fig.text(0.04, 0.5, 'Average Softmax Value (Logarithmic)', va='center', rotation='vertical', fontsize=14)

    # Adjust the spacing between subplots
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    fig.subplots_adjust(top=0.85)  # Adjust the top spacing for the main title

    # Display the plot
    plt.show()

    # Save the plot if required
    if save and filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved average softmax outputs for CIFAR10 predictions. {filename}")

# Example usage
plot_cifar10_mnist_prediction_avg_softmax(data_np_with_distances, cifar10_classes, filename="figures/cifar10_mnist_prediction_avg_softmax.png", save=True)
# Saved average softmax outputs for CIFAR10 predictions. figures/cifar10_mnist_prediction_avg_softmax.png

# TODO
# 1. Move all function definitions to CIFAR10_helper_functions.py
# 2. Add function calls to latex file referencing tables and plots
# 3. sanity check 5 random totals



