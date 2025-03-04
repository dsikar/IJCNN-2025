import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def convert_cifar10_to_mnist_format(image_array):
    """
    Converts a CIFAR10 image (32x32 RGB) to a MNIST-like format (28x28 grayscale).
    
    Steps:
      1. If RGB, convert to grayscale.
      2. Resize from 32x32 to 28x28 pixels using Lanczos resampling.
    
    Parameters:
      - image_array (numpy.ndarray): Input image as a numpy array.
    
    Returns:
      - numpy.ndarray: 28x28 grayscale image (uint8, 0-255)
    """
    # If the image is RGB, convert it to grayscale.
    if image_array.ndim == 3:
        pil_img = Image.fromarray(image_array)
        image_array = np.array(pil_img.convert('L'), dtype=np.uint8)
    # Resize from 32x32 to 28x28.
    pil_img = Image.fromarray(image_array)
    resized_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
    return np.array(resized_img, dtype=np.uint8)

def plot_mnistified_samples(dataset, class_names, conversion_fn, filename="", save_flag=False):
    """
    For each of the 10 classes, finds the first image in the dataset,
    converts it using conversion_fn (to 28x28 grayscale), and plots them in a row.
    
    Args:
        dataset: A Hugging Face dataset split (e.g., test_ds) where each item has 'img' and 'label'.
        class_names: List of class names (length 10) corresponding to labels 0-9.
        conversion_fn: Function that converts a CIFAR10 image (numpy array) to MNIST-like 28x28 format.
        filename: The filename to save the plot as an image.
        save_flag: Boolean flag to save the plot as an image if True.
    """
    # Dictionary to hold one sample per class.
    samples = {i: None for i in range(10)}
    
    # Iterate over the dataset until each class has a sample.
    for item in dataset:
        label = item['label']
        if samples[label] is None:
            image_array = np.array(item['img'])
            samples[label] = conversion_fn(image_array)
        # Break early if all classes are found.
        if all(sample is not None for sample in samples.values()):
            break

    # Create a figure with 1 row and 10 columns.
    fig, axes = plt.subplots(1, 10, figsize=(16, 2))
    fig.suptitle("MNISTified CIFAR10 examples", fontsize=16)
    for i in range(10):
        axes[i].imshow(samples[i], cmap='gray')
        axes[i].axis('off')
        # Removed: axes[i].set_title(class_names[i].upper(), fontsize=10, fontweight='bold')
        axes[i].set_xlabel(class_names[i].upper(), fontsize=10, fontweight='bold')
    plt.subplots_adjust(wspace=0.1)
    
    # Save the figure if save_flag is True.
    if save_flag:
        plt.savefig(filename)
        print(f"Plot saved as {filename}")
    plt.show()

def add_distance_to_centroid(data_np, centroids):
    """
    Add distances to centroids to the data array.
    
    Parameters:
    -----------
    data_np : numpy.ndarray
        Array containing:
        - Columns 0-9: Softmax outputs
        - Column 10: True class
        - Column 11: Predicted class
    centroids : numpy.ndarray
        Array containing centroid vectors for each class.
        
    Returns:
    --------
    numpy.ndarray
        The input array with an additional column for distance to the predicted class centroid.
    """
    # Extract softmax outputs and predicted class
    softmax_outputs = data_np[:, :10]
    predicted_classes = data_np[:, 11].astype(int)
    
    # Calculate distance to predicted class centroid for each sample
    distances = np.zeros(len(data_np))
    
    for i in range(len(data_np)):
        pred_class = predicted_classes[i]
        softmax = softmax_outputs[i]
        pred_centroid = centroids[pred_class]
        distances[i] = np.sqrt(np.sum((softmax - pred_centroid) ** 2))
    
    # Add distance column to data_np
    return np.column_stack((data_np, distances))

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
        class_distances = data_np_with_distances[data_np_with_distances[:, 11] == i, 12]
        if len(class_distances) > 0:
            min_distances[i] = np.min(class_distances)
        else:
            min_distances[i] = np.nan  # Handle case where no distances are found for a class
    
    return min_distances

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
    plt.imshow(confusion_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=3000)
    
    # Customize the plot
    plt.colorbar(label='Number of Classifications')
    plt.xlabel('MNIST Predicted Class', fontsize=14, fontweight='bold')
    plt.ylabel('CIFAR10 True Class', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix Counts For Grayscale 28x28 (MNIST Format) CIFAR10 Images', fontsize=16, fontweight='bold')
    plt.xticks(np.arange(len(cifar10_classes)), range(10), rotation=0)
    plt.yticks(np.arange(len(cifar10_classes)), cifar10_classes)
    
    # Annotate the heatmap
    for i in range(len(cifar10_classes)):
        for j in range(len(cifar10_classes)):
            plt.text(j, i, int(confusion_matrix[i, j]), ha='center', va='center', color='black', fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot if required
    if save and filename:
        plt.savefig(filename)
        print(f"Heatmap saved as {filename}")
    else:
        plt.show()

def plot_heatmap_percentages(data_np_with_distances, cifar10_classes, filename=None, save=False):
    """
    Plot a heatmap of the confusion matrix for CIFAR10 images with percentages.
    
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
    
    # Convert counts to percentages
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    confusion_matrix_percentages = np.divide(confusion_matrix, row_sums, where=row_sums != 0) * 100
    
    # Sanity check print to count the totals in the confusion matrix
    print(f"Total counts in confusion matrix: {np.sum(confusion_matrix)}")
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(confusion_matrix_percentages, cmap='Reds', aspect='auto', vmin=0, vmax=60)
    
    # Customize the plot
    plt.colorbar(label='Percentage of Classifications')
    plt.xlabel('MNIST Predicted Class', fontsize=14, fontweight='bold')
    plt.ylabel('CIFAR10 True Class', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix Percentages For Grayscale 28x28 (MNIST Format) CIFAR10 Images', fontsize=14, fontweight='bold')
    plt.xticks(np.arange(len(cifar10_classes)), range(10), rotation=0)
    plt.yticks(np.arange(len(cifar10_classes)), cifar10_classes)
    
    # Annotate the heatmap
    for i in range(len(cifar10_classes)):
        for j in range(len(cifar10_classes)):
            plt.text(j, i, f'{confusion_matrix_percentages[i, j]:.1f}%', ha='center', va='center', color='black', fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot if required
    if save and filename:
        plt.savefig(filename)
        print(f"Heatmap saved as {filename}")
    else:
        plt.show()

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