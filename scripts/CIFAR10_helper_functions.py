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

def add_distance_to_centroid(data_np, centroids):
    """
    Add distance to centroid column to data_np array.
    
    Parameters:
        data_np (numpy.ndarray): Input data array
        centroids (numpy.ndarray): Centroids array
        
    Returns:
        numpy.ndarray: Data array with added distance column
    """
    # Calculate distances and add as new column
    distances = np.linalg.norm(data_np[:, :10] - centroids[data_np[:, 10].astype(int)], axis=1)
    return np.hstack((data_np, distances.reshape(-1, 1)))

def plot_distances_to_centroids(data_np_with_distances, cifar10_classes):
    """
    Plot the distances to centroids for CIFAR10 images.
    
    Parameters:
        data_np_with_distances (numpy.ndarray): Array containing distances to centroids
        cifar10_classes (list): List of CIFAR10 class names
    """
    plt.figure(figsize=(15, 7))
    x = np.arange(len(cifar10_classes))
    width = 0.25
    
    nearest = data_np_with_distances[:, 0][:len(cifar10_classes)]
    average = data_np_with_distances[:, 1][:len(cifar10_classes)]
    furthest = data_np_with_distances[:, 2][:len(cifar10_classes)]
    
    plt.bar(x - width, nearest, width, label='Near', color='skyblue')
    plt.bar(x, average, width, label='Avg', color='lightgreen')
    plt.bar(x + width, furthest, width, label='Far', color='lightcoral')
    
    plt.xlabel('Centroid (Class)')
    plt.ylabel('Distance to Centroid')
    plt.title('Distances to Centroids for CIFAR10 Images')
    plt.legend()
    plt.xticks(x, [f'{cifar10_classes[i]} ({i})' for i in range(len(cifar10_classes))], rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

def min_distances_by_predicted_class(data_np_with_distances):
    """
    Calculate minimum distances to prediction centroids grouped by predicted class.
    
    Parameters:
        data_np_with_distances (numpy.ndarray): Array containing distances and predicted classes
    
    Returns:
        numpy.ndarray: Minimum distances for each predicted class
    """
    min_distances = np.zeros(10)
    for i in range(10):
        class_distances = data_np_with_distances[(data_np_with_distances[:, 10] == j) & (data_np_with_distances[:, 11] == i), 12]
        min_distances[i] = np.min(class_distances) if len(class_distances) > 0 else np.nan
    return min_distances

def max_distances_by_predicted_class(data_np_with_distances):
    """
    Calculate maximum distances to prediction centroids grouped by predicted class.
    
    Parameters:
        data_np_with_distances (numpy.ndarray): Array containing distances and predicted classes
    
    Returns:
        numpy.ndarray: Maximum distances for each predicted class
    """
    max_distances = np.zeros(10)
    for i in range(10):
        class_distances = data_np_with_distances[data_np_with_distances[:, 11] == i, 12]
        max_distances[i] = np.max(class_distances) if len(class_distances) > 0 else np.nan
    return max_distances

def avg_distances_by_predicted_class(data_np_with_distances):
    """
    Calculate average distances to prediction centroids grouped by predicted class.
    
    Parameters:
        data_np_with_distances (numpy.ndarray): Array containing distances and predicted classes
    
    Returns:
        numpy.ndarray: Average distances for each predicted class
    """
    avg_distances = np.zeros(10)
    for i in range(10):
        class_distances = data_np_with_distances[data_np_with_distances[:, 11] == i, 12]
        avg_distances[i] = np.mean(class_distances) if len(class_distances) > 0 else np.nan
    return avg_distances

def plot_min_max_avg_distances(min_distances, max_distances, avg_distances, cifar10_classes, filename=None, save=False):
    """
    Plot min, max and average distances to centroids for CIFAR10 images.
    
    Parameters:
        min_distances (numpy.ndarray): Minimum distances for each predicted class
        max_distances (numpy.ndarray): Maximum distances for each predicted class  
        avg_distances (numpy.ndarray): Average distances for each predicted class
        cifar10_classes (list): List of CIFAR10 class names
        filename (str): Optional filename to save plot
        save (bool): Flag to save plot
    """
    plt.figure(figsize=(15, 7))
    x = np.arange(len(cifar10_classes))
    width = 0.25
    
    plt.bar(x - width, min_distances, width, label='Min', color='skyblue')
    plt.bar(x, avg_distances, width, label='Avg', color='lightgreen')
    plt.bar(x + width, max_distances, width, label='Max', color='lightcoral')
    
    plt.xlabel('Predicted Class')
    plt.ylabel('Distance to Centroid')
    plt.title('Min, Max, and Avg Distances to MNIST Centroids for MNISTified CIFAR10 Images')
    plt.legend()
    plt.xticks(x, [f'{cifar10_classes[i]} ({i})' for i in range(len(cifar10_classes))], rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.yscale('log')
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(filename)
        print(f"Plot saved as {filename}")
    else:
        plt.show()

def plot_heatmap(data_np_with_distances, cifar10_classes, filename=None, save=False):
    """
    Plot heatmap of confusion matrix for CIFAR10 images.
    
    Parameters:
        data_np_with_distances (numpy.ndarray): Input data array
        cifar10_classes (list): List of CIFAR10 class names
        filename (str): Optional filename to save plot
        save (bool): Flag to save plot
    """
    confusion_matrix = np.zeros((len(cifar10_classes), len(cifar10_classes)))
    for i in range(len(cifar10_classes)):
        for j in range(len(cifar10_classes)):
            confusion_matrix[i, j] = np.sum((data_np_with_distances[:, 10] == i) & (data_np_with_distances[:, 11] == j))
    
    plt.figure(figsize=(12, 8))
    plt.imshow(confusion_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=3000)
    plt.colorbar(label='Number of Classifications')
    plt.xlabel('MNIST Predicted Class', fontsize=14, fontweight='bold')
    plt.ylabel('CIFAR10 True Class', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix Counts For Grayscale 28x28 (MNIST Format) CIFAR10 Images', fontsize=16, fontweight='bold')
    plt.xticks(np.arange(len(cifar10_classes)), range(10), rotation=0)
    plt.yticks(np.arange(len(cifar10_classes)), cifar10_classes)
    
    for i in range(len(cifar10_classes)):
        for j in range(len(cifar10_classes)):
            plt.text(j, i, int(confusion_matrix[i, j]), ha='center', va='center', color='black', fontsize=12)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(filename)
        print(f"Heatmap saved as {filename}")
    else:
        plt.show()

def plot_heatmap_percentages(data_np_with_distances, cifar10_classes, filename=None, save=False):
    """
    Plot heatmap of confusion matrix percentages for CIFAR10 images.
    
    Parameters:
        data_np_with_distances (numpy.ndarray): Input data array
        cifar10_classes (list): List of CIFAR10 class names
        filename (str): Optional filename to save plot
        save (bool): Flag to save plot
    """
    confusion_matrix = np.zeros((len(cifar10_classes), len(cifar10_classes)))
    for i in range(len(cifar10_classes)):
        for j in range(len(cifar10_classes)):
            confusion_matrix[i, j] = np.sum((data_np_with_distances[:, 10] == i) & (data_np_with_distances[:, 11] == j))
    
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    confusion_matrix_percentages = np.divide(confusion_matrix, row_sums, where=row_sums != 0) * 100
    
    plt.figure(figsize=(12, 8))
    plt.imshow(confusion_matrix_percentages, cmap='Reds', aspect='auto', vmin=0, vmax=60)
    plt.colorbar(label='Percentage of Classifications')
    plt.xlabel('MNIST Predicted Class', fontsize=14, fontweight='bold')
    plt.ylabel('CIFAR10 True Class', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix Percentages For Grayscale 28x28 (MNIST Format) CIFAR10 Images', fontsize=14, fontweight='bold')
    plt.xticks(np.arange(len(cifar10_classes)), range(10), rotation=0)
    plt.yticks(np.arange(len(cifar10_classes)), cifar10_classes)
    
    for i in range(len(cifar10_classes)):
        for j in range(len(cifar10_classes)):
            plt.text(j, i, f'{confusion_matrix_percentages[i, j]:.1f}%', ha='center', va='center', color='black', fontsize=12)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(filename)
        print(f"Heatmap saved as {filename}")
    else:
        plt.show()

def create_cifar_mnist_table(data_np):
    """
    Create summary table for CIFAR10 classes from data_np array.
    
    Parameters:
        data_np (numpy.ndarray): Input data array
    
    Returns:
        numpy.ndarray: Summary table with thresholds and counts
    """
    thresholds = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02])
    alpha_mask = data_np[:, 10] <= 9
    alpha_data = data_np[alpha_mask]
    predicted_classes = alpha_data[:, 11].astype(int)
    distances = alpha_data[:, 12]
    
    table_rows = []
    for thresh in thresholds:
        row = [thresh]
        class_counts = [
            np.sum((predicted_classes == d) & (distances <= thresh))
            for d in range(10)
        ]
        row.extend(class_counts)
        tbt = np.sum(distances <= thresh)
        tat = np.sum(distances > thresh)
        row.extend([tbt, tat])
        table_rows.append(row)
    
    return np.array(table_rows)

def plot_cifar10_mnist_prediction_avg_softmax(data_np_with_distances, cifar10_classes, filename=None, save=False):
    """
    Plot average softmax values for CIFAR10 images grouped by predicted class.
    
    Parameters:
        data_np_with_distances (numpy.ndarray): Input data array
        cifar10_classes (list): List of CIFAR10 class names
        filename (str): Optional filename to save plot
        save (bool): Flag to save plot
    """
    labels = np.unique(data_np_with_distances[:, 11]).astype(int)
    fig, axs = plt.subplots(1, len(labels), figsize=(20, 5))
    fig.suptitle('Softmax Average Distributions for MNISTified CIFAR10 Predictions', fontsize=16)

    for i, label in enumerate(labels):
        class_predictions = data_np_with_distances[data_np_with_distances[:, 11] == label, :10]
        averages = np.mean(class_predictions, axis=0)
        axs[i].bar(np.arange(10), averages, color='lightcoral')
        axs[i].set_yscale('log')
        axs[i].set_ylim(bottom=1e-4)
        axs[i].set_title(f'{cifar10_classes[label].capitalize()} ({label})', fontsize=12)
        axs[i].set_xticks(np.arange(10))
        axs[i].set_xticklabels(np.arange(10), fontsize=10)
        axs[i].set_xticks(np.arange(10), minor=True)
        axs[i].grid(True, which='minor', axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
        axs[i].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    fig.text(0.5, 0.04, 'Class Index', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'Average Softmax Value (Logarithmic)', va='center', rotation='vertical', fontsize=14)
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    fig.subplots_adjust(top=0.85)

    if save and filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved average softmax outputs for CIFAR10 predictions. {filename}")

def plot_mnistified_samples(dataset, class_names, conversion_fn, filename="", save_flag=False):
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
