import numpy as np
import os
import matplotlib.pyplot as plt

# COPIED FROM /home/daniel/git/work-in-progress/scripts/mnist_helper_functions.py

def read_bytes_from_file(file_path, num_bytes, start=0, verbose=False):
    """
    Reads a specified number of bytes from a file starting at a given offset and prints them in hexadecimal.

    Parameters:
    - file_path: The path to the file.
    - num_bytes: The number of bytes to read from the file.
    - start: The offset from the beginning of the file to start reading bytes (default is 0).
    """
        
    with open(file_path, 'rb') as file:
        file.seek(start)  # Move to the start position
        data = file.read(num_bytes)
        if verbose:
          print("Hexadecimal representation of", num_bytes, "bytes starting from byte", start, ":")
          print(data.hex())
    return data.hex()

def hex_to_numpy_array(hex_data, row_length):
    """
    Converts a hexadecimal string into a numpy array of specified row length.

    Parameters:
    - hex_data: Hexadecimal string to be converted.
    - row_length: The length of each row in the resulting array.

    Returns:
    - A numpy array representing the hexadecimal data.

    Example:
    # Assuming read_bytes_from_file has been called and hex_data is obtained
    file_path = 'data/MNIST/raw/train-images-idx3-ubyte'
    num_bytes = 784
    start=16
    verbose = False
    image1 = read_bytes_from_file(file_path, num_bytes, start, verbose)
    hex_data = image1  # This is a placeholder. Use actual hex data from read_bytes_from_file
    row_length = 28  # For MNIST images

    try:
        image_array = hex_to_numpy_array(hex_data, row_length)
        print("Numpy array shape:", image_array.shape)
    except ValueError as e:
        print(e)
    """
    # Convert hex_data to bytes in decimal format
    byte_data = bytes.fromhex(hex_data)

    # Calculate the total number of expected rows
    total_bytes = len(byte_data)
    if total_bytes % row_length != 0:
        raise ValueError("The total number of bytes is not evenly divisible by the specified row length.")

    # Calculate the number of rows
    num_rows = total_bytes // row_length

    # Convert byte data to a numpy array and reshape
    np_array = np.frombuffer(byte_data, dtype=np.uint8).reshape((num_rows, row_length))

    return np_array

def get_mnist_img_array(filename, index, verbose = False):
    """
    Get image from file at a given index

    Parameters
    ==========
    filename: string, the mnist binary file
    index: the index to display
    verbose: boolean, display debug info

    Example
    =========
    pmnist_img = 'data/MNIST/raw/train-images-idx3-ubyte'  # perturbed-train-images-idx3-ubyte
    index = 0
    verbose = False
    get_mnist_img_array(pmnist_img, index, verbose)

    Note
    =========
    Create a symlink to the MNIST dataset in the data directory e.g.:
    $ ln -s /home/daniel/git/work-in-progress/scripts/data/MNIST/ /home/daniel/git/IJCNN-2025/scripts/data/MNIST
    """
    if filename == 'train':
        filename = 'scripts/data/MNIST/raw/train-images-idx3-ubyte'
    elif filename == 'test':
        filename = 'scripts/data/MNIST/raw/t10k-images-idx3-ubyte'
    elif filename ==  'perturbed':
        filename = 'scripts/data/MNIST/raw/t1210k-perturbation-levels-idx0-ubyte'
    else:
        # default to train
        filename = 'data/MNIST/raw/train-images-idx3-ubyte'
    
    num_bytes = 784 #(28x28)
    start=16+(index*num_bytes)
    row_length = 28
    file_size = os.path.getsize(filename)
    image_array = None
    if file_size < start+num_bytes:
        print("The specified image index {} is out of bounds for the file.".format(index))
    else: 
        img_hex = read_bytes_from_file(filename, num_bytes, start, verbose)
        image_array = hex_to_numpy_array(img_hex, row_length)
    return image_array  

def plot_image(image, title):
    """
    Display a grayscale image with the specified title.
    
    Parameters:
    -----------
    image : numpy.ndarray
        The image array to display
    title : str
        The title to display above the image
    """
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# END COPY
    
def calculate_distances(train_data, centroids):
    """
    Calculate Euclidean distances between softmax outputs and centroids for true and predicted labels.
    
    Parameters:
    -----------
    train_data : numpy.ndarray
        Array of shape (n_samples, 12) containing:
        - Columns 0-9: Softmax outputs
        - Column 10: True class
        - Column 11: Predicted class
    centroids : numpy.ndarray
        Array of shape (10, 10) containing centroid vectors for each class
        
    Returns:
    --------
    numpy.ndarray
        Original train_data with two new columns appended:
        - Column 12: Euclidean distance to true class centroid
        - Column 13: Euclidean distance to predicted class centroid
    """
    # Extract softmax outputs
    softmax_outputs = train_data[:, :10]
    
    # Extract true and predicted labels
    true_labels = train_data[:, 10].astype(int)
    pred_labels = train_data[:, 11].astype(int)
    
    # Initialize arrays to store distances
    true_distances = np.zeros(len(train_data))
    pred_distances = np.zeros(len(train_data))
    
    # Calculate Euclidean distances
    for i in range(len(train_data)):
        # Get corresponding centroids
        true_centroid = centroids[true_labels[i]]
        pred_centroid = centroids[pred_labels[i]]
        
        # Calculate distances
        true_distances[i] = np.sqrt(np.sum((softmax_outputs[i] - true_centroid) ** 2))
        pred_distances[i] = np.sqrt(np.sum((softmax_outputs[i] - pred_centroid) ** 2))
    
    # Append distances to train_data
    return np.column_stack((train_data, true_distances, pred_distances))

def find_thresholds(train_data_with_distances, same_class=True):
    """
    Find distance thresholds for each class based on misclassified points.
    
    Parameters:
    -----------
    train_data_with_distances : numpy.ndarray
        Array containing:
        - Columns 0-9: Softmax outputs
        - Column 10: True class
        - Column 11: Predicted class
        - Column 12: Distance to true class centroid
        - Column 13: Distance to predicted class centroid
    same_class : bool, optional
        If True, find thresholds for points that belong to class i but were misclassified
        If False, find thresholds for points that were incorrectly classified as class i
        
    Returns:
    --------
    tuple
        (thresholds, threshold_indexes)
        - thresholds: numpy array of minimum distances for each class
        - threshold_indexes: indexes in original array where minimum distances were found
    """
    thresholds = np.zeros(10)
    threshold_indexes = np.zeros(10)
    
    for i in range(10):
        if same_class:
            # Find points that belong to class i but were misclassified
            misclassified_mask = (train_data_with_distances[:, 10] == i) & \
                               (train_data_with_distances[:, 11] != i)
            distance_column = 12  # Use distance to true class centroid
        else:
            # Find points that were incorrectly classified as class i
            misclassified_mask = (train_data_with_distances[:, 10] != i) & \
                               (train_data_with_distances[:, 11] == i) & \
                               (train_data_with_distances[:, 10] != train_data_with_distances[:, 11])
            distance_column = 13  # Use distance to predicted class centroid
            
        misclassified_points = train_data_with_distances[misclassified_mask]
        
        print(f"Class {i}: Number of misclassified points: {len(misclassified_points)}")
        
        if len(misclassified_points) > 0:
            # Use pre-calculated distances
            distances = misclassified_points[:, distance_column]
            min_distance_idx = np.argmin(distances)
            
            thresholds[i] = distances[min_distance_idx]
            # Map the index back to the original array
            original_index = np.where(misclassified_mask)[0][min_distance_idx]
            threshold_indexes[i] = original_index
            
            print(f"Class {i}: Minimum distance: {thresholds[i]}")
            print(f"Class {i}: Original index of the nearest misclassified point: {original_index}")
        else:
            thresholds[i] = np.inf
            threshold_indexes[i] = -1
            print(f"Class {i}: No misclassified points found")
    
    return thresholds, threshold_indexes

def find_thresholds_digits_only(train_data_with_distances, same_class=True):
    """
    Find distance thresholds for each digit class (0-9) based on misclassified points,
    considering only numeric characters (true labels 0-9).
    
    Parameters:
    -----------
    train_data_with_distances : numpy.ndarray
        Array containing:
        - Columns 0-9: Softmax outputs
        - Column 10: True class
        - Column 11: Predicted class
        - Column 12: Distance to true class centroid
        - Column 13: Distance to predicted class centroid
    same_class : bool, optional
        If True, find thresholds for points that belong to class i but were misclassified
        If False, find thresholds for points that were incorrectly classified as class i
    
    Returns:
    --------
    tuple
        (thresholds, threshold_indexes)
        - thresholds: numpy array of minimum distances for each class
        - threshold_indexes: indexes in original array where minimum distances were found
    """
    # Filter to only include numeric characters (true labels 0-9)
    digits_mask = (train_data_with_distances[:, 10] < 10)
    digits_data = train_data_with_distances[digits_mask]
    
    # Store original indices for mapping back later
    original_indices = np.where(digits_mask)[0]
    
    thresholds = np.zeros(10)
    threshold_indexes = np.zeros(10, dtype=int)
    
    for i in range(10):
        if same_class:
            # Find points that belong to digit i but were misclassified
            misclassified_mask = (digits_data[:, 10] == i) & \
                                (digits_data[:, 11] != i)
            distance_column = 12  # Use distance to true class centroid
        else:
            # Find points that were incorrectly classified as digit i
            misclassified_mask = (digits_data[:, 10] != i) & \
                                (digits_data[:, 11] == i) & \
                                (digits_data[:, 10] != digits_data[:, 11])
            distance_column = 13  # Use distance to predicted class centroid
        
        misclassified_points = digits_data[misclassified_mask]
        print(f"Digit {i}: Number of misclassified points: {len(misclassified_points)}")
        
        if len(misclassified_points) > 0:
            # Use pre-calculated distances
            distances = misclassified_points[:, distance_column]
            min_distance_idx = np.argmin(distances)
            thresholds[i] = distances[min_distance_idx]
            
            # Map the index back to the original array using our stored indices
            original_index = original_indices[np.where(misclassified_mask)[0][min_distance_idx]]
            threshold_indexes[i] = original_index
            
            print(f"Digit {i}: Minimum distance: {thresholds[i]}")
            print(f"Digit {i}: Original index of the nearest misclassified point: {original_index}")
        else:
            thresholds[i] = np.inf
            threshold_indexes[i] = -1
            print(f"Digit {i}: No misclassified points found")
    
    return thresholds, threshold_indexes

def calculate_entropy(probs):
    """Calculate entropy for a single probability distribution"""
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1.0)
    return -np.sum(probs * np.log2(probs))

def debug_point(index, train_data_with_distances, thresholds, threshold_indexes, label="Same class"):
    """
    Print threshold, index, data values and entropy for a given class index.
    
    Parameters:
    -----------
    index : int
        Class index to debug
    train_data_with_distances : numpy.ndarray
        Array containing the data
    thresholds : numpy.ndarray
        Array of thresholds
    threshold_indexes : numpy.ndarray
        Array of threshold indexes
    label : str, optional
        Label to identify what case we're debugging (default: "Same class")
    """
    print(f"\n=== Debugging {label} ===")
    print(f"threshold[{index}]")
    print(thresholds[index])
    print(f"index[{index}]")
    print(threshold_indexes[index])
    point_idx = int(threshold_indexes[index])
    print(f"train_data_with_distances[{point_idx}]")
    print(train_data_with_distances[point_idx])
    print(f"entropy:")
    print(calculate_entropy(train_data_with_distances[point_idx, :10]))

import matplotlib.pyplot as plt
import numpy as np

def plot_softmax(train_data_with_distances, index):
    """
    Plot softmax output for a given index as a bar chart with logarithmic scale.
    
    Parameters:
    -----------
    train_data_with_distances : numpy.ndarray
        Array containing the data
    index : int
        Index of the point to plot
    """
    # Get the data point
    point = train_data_with_distances[index]
    
    # Extract softmax outputs and classes
    softmax_output = point[:10]
    true_class = int(point[10])
    predicted_class = int(point[11])
    
    # Handle zero values for log scale (replace with small value)
    epsilon = 1e-15
    softmax_output = np.clip(softmax_output, epsilon, 1.0)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), softmax_output)
    
    # Set log scale for y-axis
    plt.yscale('log')
    
    # Customize the plot
    plt.title(f'Softmax output (log scale), true class: {true_class}, predicted class: {predicted_class}')
    plt.xlabel('Class')
    plt.ylabel('Probability (log scale)')
    plt.xticks(range(10))
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.show()

def plot_softmax_comparison(train_data_with_distances, same_class_idx, different_class_idx):
    """
    Plot softmax outputs for two points side by side with logarithmic scale and their entropy comparison.
        
    Parameters:
    -----------
    train_data_with_distances : numpy.ndarray
        Array of shape (n_samples, 14) containing:
        - Columns 0-9: Softmax outputs (class probabilities)
        - Column 10: True class label
        - Column 11: Predicted class label
        - Column 12: Distance to true class centroid
        - Column 13: Distance to predicted class centroid
    same_class_idx : int
        Index for the same class case
    different_class_idx : int
        Index for the different class case
    """
    # Create figure with three subplots, making the entropy plot narrower
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), 
                                       gridspec_kw={'width_ratios': [2, 2, 1]})
    
    # Handle zero values for log scale
    epsilon = 1e-15
    
    # Plot Same Class case
    point1 = train_data_with_distances[same_class_idx]
    softmax1 = np.clip(point1[:10], epsilon, 1.0)
    true_class1 = int(point1[10])
    pred_class1 = int(point1[11])
    
    ax1.bar(range(10), softmax1)
    ax1.set_yscale('log')
    ax1.set_title(f'Same Class (idx={same_class_idx})\nTrue: {true_class1}, Pred: {pred_class1}')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Probability (log scale)')
    ax1.set_xticks(range(10))
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Plot Different Class case
    point2 = train_data_with_distances[different_class_idx]
    softmax2 = np.clip(point2[:10], epsilon, 1.0)
    true_class2 = int(point2[10])
    pred_class2 = int(point2[11])
    
    ax2.bar(range(10), softmax2)
    ax2.set_yscale('log')
    ax2.set_title(f'Different Class (idx={different_class_idx})\nTrue: {true_class2}, Pred: {pred_class2}')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Probability (log scale)')
    ax2.set_xticks(range(10))
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Calculate entropy for both distributions
    def calculate_entropy(probs):
        return -np.sum(probs * np.log2(probs))
    
    entropy1 = calculate_entropy(softmax1)
    entropy2 = calculate_entropy(softmax2)
    
    # Plot entropy comparison
    entropy_data = [entropy1, entropy2]
    labels = [f'Same Class\n(idx={same_class_idx})', f'Different Class\n(idx={different_class_idx})']
    
    ax3.bar(labels, entropy_data)
    ax3.set_yscale('log')
    ax3.set_title('Entropy Comparison\n(log scale)')
    ax3.set_ylabel('Entropy (bits)')
    ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add entropy values as text annotations
    for i, entropy in enumerate(entropy_data):
        ax3.text(i, entropy, f'{entropy:.3f}', 
                ha='center', va='bottom')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix(data, digits_only=False):
    """
    Compute confusion matrix from data.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Array containing:
        - Column 10: True class
        - Column 11: Predicted class
    digits_only : bool, optional
        If True, only consider data points with true class 0-9
        
    Returns:
    --------
    numpy.ndarray
        10x10 confusion matrix where rows are true classes and columns are predicted classes
    """
    # Initialize a 10x10 confusion matrix for digits
    conf_matrix = np.zeros((10, 10), dtype=int)
    
    if digits_only:
        # Filter to only include data points with true class 0-9
        mask = (data[:, 10] < 10)
        filtered_data = data[mask]
    else:
        # Use all data
        filtered_data = data
    
    # Fill in the confusion matrix
    for i in range(len(filtered_data)):
        true_class = int(filtered_data[i, 10])
        pred_class = int(filtered_data[i, 11])
        
        # Only count entries where true_class is a digit (0-9)
        if true_class < 10:
            conf_matrix[true_class, pred_class] += 1
    
    return conf_matrix

def plot_confusion_matrix(conf_matrix, prefix="", filename=None, save=False):
    """
    Plot a confusion matrix.
    
    Parameters:
    -----------
    conf_matrix : numpy.ndarray
        10x10 confusion matrix where rows are true classes and columns are predicted classes
    prefix : str, optional
        Prefix for the plot title
    filename : str, optional
        Filename to save the plot
    save : bool, optional
        If True, save the plot to a file
        
    Returns:
    --------
    None
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', 
                xticklabels=range(10), yticklabels=range(10))
    
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(f'{prefix}Confusion Matrix')
    
    if save and filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix as {filename}")
    
    plt.tight_layout()
    plt.show()

def find_thresholds_alphabetic_only(train_data_with_distances):
    """
    Find distance thresholds for each predicted digit class based on alphabetic characters.
    
    Parameters:
    -----------
    train_data_with_distances : numpy.ndarray
        Array containing:
        - Columns 0-9: Softmax outputs
        - Column 10: True class
        - Column 11: Predicted class
        - Column 12: Distance to true class centroid (NaN for alphabetic)
        - Column 13: Distance to predicted class centroid
    
    Returns:
    --------
    tuple
        (thresholds, threshold_indexes, avg_distances)
        - thresholds: numpy array of minimum distances for each digit class
        - threshold_indexes: indexes in original array where minimum distances were found
        - avg_distances: average distance for alphabetic chars predicted as each digit
    """
    # Filter to only include alphabetic characters (true labels 10-61)
    alpha_mask = (train_data_with_distances[:, 10] >= 10) & (train_data_with_distances[:, 10] <= 61)
    alpha_data = train_data_with_distances[alpha_mask]
    
    # Store original indices for mapping back later
    original_indices = np.where(alpha_mask)[0]
    
    thresholds = np.zeros(10)
    threshold_indexes = np.zeros(10, dtype=int)
    avg_distances = np.zeros(10)
    counts = np.zeros(10)
    
    for i in range(10):
        # Find alphabetic characters predicted as digit i
        pred_as_digit_mask = (alpha_data[:, 11] == i)
        points_pred_as_digit = alpha_data[pred_as_digit_mask]
        
        print(f"Digit {i}: Number of alphabetic characters predicted as this digit: {len(points_pred_as_digit)}")
        
        if len(points_pred_as_digit) > 0:
            # Use distance to predicted class centroid (column 13)
            distances = points_pred_as_digit[:, 13]
            
            # Find minimum distance
            min_distance_idx = np.argmin(distances)
            thresholds[i] = distances[min_distance_idx]
            
            # Map the index back to the original array
            original_index = original_indices[np.where(pred_as_digit_mask)[0][min_distance_idx]]
            threshold_indexes[i] = original_index
            
            # Calculate average distance
            avg_distances[i] = np.mean(distances)
            counts[i] = len(distances)
            
            print(f"Digit {i}: Minimum distance: {thresholds[i]}")
            print(f"Digit {i}: Average distance: {avg_distances[i]}")
            print(f"Digit {i}: Original index of the alphabetic character with min distance: {original_index}")
            
            # Print the true class (letter) of the closest point
            true_class = int(points_pred_as_digit[min_distance_idx, 10])
            letter = get_letter_from_class(true_class)
            print(f"Digit {i}: Closest alphabetic character is '{letter}' (class {true_class})")
        else:
            thresholds[i] = np.inf
            threshold_indexes[i] = -1
            avg_distances[i] = np.nan
            print(f"Digit {i}: No alphabetic characters predicted as this digit")
    
    return thresholds, threshold_indexes, avg_distances

def get_letter_from_class(class_num):
    """
    Convert class number to corresponding letter.
    
    Parameters:
    -----------
    class_num : int
        Class number (10-61)
    
    Returns:
    --------
    str
        Corresponding letter (A-Z, a-z)
    """
    if 10 <= class_num <= 35:
        # Uppercase letters (A-Z)
        return chr(class_num - 10 + ord('A'))
    elif 36 <= class_num <= 61:
        # Lowercase letters (a-z)
        return chr(class_num - 36 + ord('a'))
    else:
        return f"Not a letter (class {class_num})"

def plot_thresholds_comparison(thresholds_same, thresholds_different, prefix="", filename="", save=False):
    """
    Plot threshold pairs for same and different cases side by side.
    
    Parameters:
    -----------
    thresholds_same : numpy.ndarray
        Array of thresholds for same class
    thresholds_different : numpy.ndarray
        Array of thresholds for different class
    """
    # Create figure and axis
    plt.figure(figsize=(12, 5))
    
    # Set the positions for the bars
    x = np.arange(len(thresholds_same))
    width = 0.35
    
    # Create the bars
    plt.bar(x - width/2, thresholds_same, width, label='Same Class', color='skyblue')
    plt.bar(x + width/2, thresholds_different, width, label='Different Class', color='lightcoral')
    
    # Customize the plot
    plt.yscale('log')  # Use log scale since values are very different
    plt.xlabel('Index')
    plt.ylabel('Threshold Value (log scale)')
    plt.title("{}Threshold Comparison: True misclassifed vs Predicted misclassified".format(prefix))
    plt.legend()

        # Move legend to bottom right
    plt.legend(loc='lower right')
    
    # Set x-ticks at the center of each pair
    plt.xticks(x, range(10))
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value annotations
    for i in range(len(thresholds_same)):
        # Annotate Same Class threshold
        plt.text(i - width/2, thresholds_same[i], f'{thresholds_same[i]:.3f}', 
                ha='center', va='bottom', rotation=0)
        # Annotate Different Class threshold
        plt.text(i + width/2, thresholds_different[i], f'{thresholds_different[i]:.3f}', 
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.show()    

    if save:
        plt.savefig(filename)

import numpy as np
import matplotlib.pyplot as plt

def plot_thresholds_comparison_canonical_x_axis(thresholds_same, thresholds_different, dataset="mnist", prefix="", filename="", save=False):
    """
    Plot threshold pairs for same and different cases side by side with dataset-specific labels.
    
    Parameters:
    -----------
    thresholds_same : numpy.ndarray
        Array of thresholds for same class
    thresholds_different : numpy.ndarray
        Array of thresholds for different class
    dataset : str
        Dataset name ('mnist' or 'cifar10') to determine x-axis labels
    prefix : str
        Prefix for the plot title
    filename : str
        File name to save the plot (if save is True)
    save : bool
        Whether to save the plot to a file
    """
    # Define class labels for each dataset
    if dataset.lower() == "mnist":
        class_labels = [str(i) for i in range(10)]  # Digits 0-9
    elif dataset.lower() == "cifar10":
        class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR-10 labels
    else:
        raise ValueError("Dataset must be 'mnist' or 'cifar10'")

    # Create figure and axis
    plt.figure(figsize=(12, 5))
    
    # Set the positions for the bars
    x = np.arange(len(thresholds_same))
    width = 0.35
    
    # Create the bars
    plt.bar(x - width/2, thresholds_same, width, label='Same Class', color='skyblue')
    plt.bar(x + width/2, thresholds_different, width, label='Different Class', color='lightcoral')
    
    # Customize the plot
    plt.yscale('log')  # Use log scale since values are very different
    plt.xlabel('Class')
    plt.ylabel('Threshold Value (log scale)')
    plt.title(f"{prefix} Threshold Comparison: True misclassified vs Predicted misclassified")
    plt.legend()

    # Move legend to bottom right
    plt.legend(loc='lower right')
    
    # Set x-ticks with dataset-specific labels
    plt.xticks(x, class_labels, rotation=45 if dataset.lower() == "cifar10" else 0)
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value annotations
    for i in range(len(thresholds_same)):
        # Annotate Same Class threshold
        plt.text(i - width/2, thresholds_same[i], f'{thresholds_same[i]:.3f}', 
                 ha='center', va='bottom', rotation=0)
        # Annotate Different Class threshold
        plt.text(i + width/2, thresholds_different[i], f'{thresholds_different[i]:.3f}', 
                 ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.show()    

    if save:
        plt.savefig(filename)
        plt.close()

def plot_alphabetic_distances(min_distances, avg_distances, prefix="", filename="", save=False):
    """
    Plot bar charts for average and minimum distances of alphabetic characters to digit centroids.
    
    Parameters:
    -----------
    min_distances : numpy.ndarray
        Array of minimum distances for alphabetic characters to each digit
    avg_distances : numpy.ndarray
        Array of average distances for alphabetic characters to each digit
    prefix : str, optional
        Prefix for the plot title
    filename : str, optional
        Filename to save the plot
    save : bool, optional
        If True, save the plot to a file
    """
    # Create figure and axis
    plt.figure(figsize=(12, 5))
    
    # Set the positions for the bars
    x = np.arange(len(min_distances))
    width = 0.35
    
    # Create the bars
    plt.bar(x - width/2, avg_distances, width, label='Average Distance', color='green')
    plt.bar(x + width/2, min_distances, width, label='Minimum Distance', color='cyan')
    
    # Customize the plot
    plt.yscale('log')  # Use log scale since values may be very different
    plt.xlabel('Digit')
    plt.ylabel('Distance Value (log scale)')
    plt.title(f"{prefix}Distances of Alphabetic Characters to Digit Centroids")
    
    # Move legend to bottom right
    plt.legend(loc='lower right')
    
    # Set x-ticks at the center of each pair
    plt.xticks(x, range(10))
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value annotations
    for i in range(len(min_distances)):
        # Skip NaN values (if any)
        if np.isfinite(avg_distances[i]):
            # Annotate Average Distance
            plt.text(i - width/2, avg_distances[i], f'{avg_distances[i]:.3f}',
                    ha='center', va='bottom', rotation=0)
        
        if np.isfinite(min_distances[i]):
            # Annotate Minimum Distance
            plt.text(i + width/2, min_distances[i], f'{min_distances[i]:.3f}',
                    ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    
    if save and filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved distance comparison plot as {filename}")
    
    plt.show()

def load_image_as_array(directory_path, filename):
    """
    Loads a PNG image from a specified directory and filename, returning it as a NumPy array.
    
    Parameters:
    - directory_path (str): Path to the directory containing the image
    - filename (str): Name of the PNG file (e.g., 'img001-001.png')
    
    Returns:
    - numpy.ndarray: Grayscale image as a 2D NumPy array (uint8, 0–255)
    
    Raises:
    - FileNotFoundError: If the image file does not exist
    - ValueError: If the file is not a valid PNG or cannot be loaded
    """
    try:
        # Construct the full path to the image
        full_path = os.path.join(directory_path, filename)
        
        # Check if the file exists
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")
        if not filename.lower().endswith('.png'):
            raise ValueError(f"File must be a PNG: {filename}")
        
        # Open the image using PIL
        img = Image.open(full_path)
        
        # Convert to RGB if not already, then to grayscale ('L' mode)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        gray_img = img.convert('L')  # Grayscale conversion
        
        # Convert to NumPy array (uint8, 0–255)
        image_array = np.array(gray_img, dtype=np.uint8)
        
        # Print for debugging
        print(f"Loaded image '{filename}' - Shape: {image_array.shape}, "
              f"Min/Max Pixel: {image_array.min()}/{image_array.max()}")
        
        return image_array
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading image: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading image: {e}")
        return None

from PIL import Image

def convert_to_mnist_format_preserve_aspect(image_array, threshold=128):
    """
    Converts an input handwritten character image to MNIST format by:
    1. Cropping excess whitespace
    2. Resizing to preserve aspect ratio so the longer side is 22px
    3. Inverting colors (character becomes white on black background)
    4. Padding to create a standardized 28x28 pixel image
    
    Parameters:
    - image_array (numpy.ndarray): 2D grayscale image array (uint8, 0-255)
    - threshold (int): Pixel threshold to detect character vs background (default: 128)
    
    Returns:
    - numpy.ndarray: MNIST-formatted 28x28 grayscale image array (uint8, 0-255)
    """
    # Validate input
    if not isinstance(image_array, np.ndarray):
        raise ValueError("Input must be a NumPy array")
    if len(image_array.shape) != 2:
        raise ValueError("Input must be a 2D grayscale image array")
    
    # STEP 1: Crop excess whitespace
    h, w = image_array.shape
    rows, cols = np.where(image_array < threshold)
    
    # If no pixel is found below threshold, return a blank 28x28 image
    if rows.size == 0 or cols.size == 0:
        return np.ones((28, 28), dtype=np.uint8) * 255
    
    left_index = np.min(cols)
    right_index = np.max(cols)
    top_index = np.min(rows)
    bottom_index = np.max(rows)
    
    left = max(0, int(left_index))
    right = min(w, int(right_index + 1))
    top = max(0, int(top_index))
    bottom = min(h, int(bottom_index + 1))
    
    if left >= right or top >= bottom:
        return np.ones((28, 28), dtype=np.uint8) * 255
    
    cropped_image = image_array[top:bottom, left:right]
    
    # STEP 2: Resize while preserving aspect ratio
    cropped_h, cropped_w = cropped_image.shape
    max_dim = max(cropped_h, cropped_w)
    # Scale so the longer dimension is 22
    scale = 22 / max_dim  
    new_w = max(1, int(round(cropped_w * scale)))
    new_h = max(1, int(round(cropped_h * scale)))
    
    pil_img = Image.fromarray(cropped_image, mode='L')
    resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_array = np.array(resized_img, dtype=np.uint8)
    
    # STEP 3: Invert colors (white on black background)
    inverted_array = 255 - resized_array
    
    # STEP 4: Pad to 28x28
    padded_28x28 = np.zeros((28, 28), dtype=np.uint8)  # black background
    
    # Center the resized image in the 28x28 canvas
    start_row = (28 - new_h) // 2
    start_col = (28 - new_w) // 2
    padded_28x28[start_row:start_row+new_h, start_col:start_col+new_w] = inverted_array
    
    return padded_28x28

def plot_closest_letters_to_digits(alpha_indexes, directory, png_files, alpha_min_distances, alpha_avg_distances, data_np, filename="", save=False):
    """
    Plot the closest alphabetic character to each digit, along with distance information.
    
    Parameters:
    -----------
    alpha_indexes : numpy.ndarray
        Array of indexes of the closest alphabetic character to each digit
    directory : str
        Directory containing the image files
    png_files : list
        List of PNG filenames
    alpha_min_distances : numpy.ndarray
        Array of minimum distances for alphabetic characters to each digit
    alpha_avg_distances : numpy.ndarray
        Array of average distances for alphabetic characters to each digit
    data_np : numpy.ndarray
        Array containing the data
        filename : str, optional
        Filename to save the plot
    save : bool, optional
        If True, save the plot to a file     
    """
    plt.figure(figsize=(20, 5))
    
    for digit in range(10):
        plt.subplot(1, 10, digit + 1)
        
        # Get the index of the closest alphabetic character to this digit
        closest_idx = int(alpha_indexes[digit])
        
        if closest_idx >= 0:  # Valid index (not -1)
            # Load and display the image
            image_array = load_image_as_array(directory, png_files[closest_idx])
            mnist_array = convert_to_mnist_format_preserve_aspect(image_array, threshold=128)
            plt.imshow(mnist_array, cmap='gray')
            
            # Get the true class label from the data
            true_class = int(data_np[closest_idx, 10])
            letter = get_letter_from_class(true_class)
            
            # Create the information text
            min_dist = alpha_min_distances[digit]
            avg_dist = alpha_avg_distances[digit]
            
            info_text = f"Digit: {digit}\nLetter: {letter}\nMin dist: {min_dist:.5f}\nAvg dist: {avg_dist:.5f}"
            plt.title(info_text, fontsize=10)
        else:
            # No closest letter found
            plt.text(0.5, 0.5, f"Digit {digit}:\nNo letter found", 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.axis('off')
    
    plt.suptitle('Closest Alphabetic Characters to Each Digit', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust to make room for the main title
    plt.tight_layout()
    plt.show()

    if save and filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved distance comparison plot as {filename}")   

import os

def get_png_files(directory_path):
    """
    Returns a list of all .png file names in the specified directory.
    
    Parameters:
    - directory_path (str): Path to the directory to search for PNG files
    
    Returns:
    - list: List of strings, each representing a .png filename in the directory
    
    Raises:
    - FileNotFoundError: If the directory does not exist
    - NotADirectoryError: If the path is not a directory
    """
    try:
        # Check if the path exists and is a directory
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Path is not a directory: {directory_path}")
        
        # List to store PNG filenames
        png_files = []
        
        # Iterate through all files in the directory
        for filename in os.listdir(directory_path):
            # Check if the file ends with .png (case-insensitive)
            if filename.lower().endswith('.png'):
                png_files.append(filename)
        
        # Sort the list for consistency (optional)
        png_files.sort()
        
        return png_files
    
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}")
        return []

## Threshold table
## Legacy function
    
def generate_threshold_table_rows(train_data_with_distances, thresholds_same, thresholds_different):
    """
    Generate rows of the threshold table including both T1 and T2 threshold statistics
    
    Parameters:
    -----------
    train_data_with_distances : numpy.ndarray
        Array containing the data where:
        - Columns 0-9: Softmax outputs
        - Column 10: True class
        - Column 11: Predicted class
        - Column 12: Distance to true class centroid
        - Column 13: Distance to predicted class centroid
    thresholds_same : numpy.ndarray
        Array of 10 threshold values (T1) for true class
    thresholds_different : numpy.ndarray
        Array of 10 threshold values (T2) for stricter classification
    """
    # Initialize the LaTeX table string
    latex_str = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|}\n\\hline\n"
    latex_str += "\\multicolumn{11}{|c|}{Thresholds} \\\\\n\\hline\n"
    latex_str += "Digit & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 \\\\\n\\hline\n"
    
    # Initialize arrays to store statistics
    totals = []
    corrects = []
    percentages = []
    below_threshold_t1 = []
    below_threshold_t1_pct = []
    below_threshold_t2 = []
    below_threshold_t2_pct = []
    misclass_below_t1 = []  # New array for misclassification counts

    # For each digit
    for digit in range(10):
        # Get mask for this digit as true class
        digit_mask = train_data_with_distances[:, 10] == digit
        digit_samples = train_data_with_distances[digit_mask]
        
        # Count total and correct classifications
        total = len(digit_samples)
        correct_mask = digit_samples[:, 11] == digit_samples[:, 10]
        correct = np.sum(correct_mask)
        
        # Calculate percentage
        percentage = (correct / total) * 100
        
        # Get samples below T1 threshold
        # Note: thresholds_same
        below_t1_count = np.sum(digit_samples[:, 12] < thresholds_same[digit])
        below_t1_percentage = (below_t1_count / total) * 100
        
        # Get correctly classified samples below T2 threshold
        correct_samples = digit_samples[correct_mask]
        below_t2_count = np.sum(correct_samples[:, 12] < thresholds_different[digit])
        below_t2_percentage = (below_t2_count / total) * 100
        
        # Calculate misclassifications below T1
        # Get samples where predicted class is digit but true class isn't
        misclass_mask = (train_data_with_distances[:, 11] == digit) & (train_data_with_distances[:, 10] != digit)
        misclassified_samples = train_data_with_distances[misclass_mask]
        # Count how many of these misclassified samples are below T1 threshold
        # Use distance to predicted centroid (column 13)        
        misclass_below_t1_count = np.sum(misclassified_samples[:, 13] < thresholds_same[digit])
               

        totals.append(total)
        corrects.append(correct)
        percentages.append(percentage)
        below_threshold_t1.append(below_t1_count)
        below_threshold_t1_pct.append(below_t1_percentage)
        below_threshold_t2.append(below_t2_count)
        below_threshold_t2_pct.append(below_t2_percentage)
        misclass_below_t1.append(misclass_below_t1_count)
    
    # Add rows to table
    latex_str += "Total & " + " & ".join(f"{t}" for t in totals) + " \\\\\n\\hline\n"
    latex_str += "Correct & " + " & ".join(f"{c}" for c in corrects) + " \\\\\n\\hline\n"
    latex_str += "Percentage & " + " & ".join(f"{p:.1f}\\%" for p in percentages) + " \\\\\n\\hline\n"
    latex_str += "Below T1 & " + " & ".join(f"{b}" for b in below_threshold_t1) + " \\\\\n\\hline\n"
    latex_str += "< T1 Correct \\% & " + " & ".join(f"{p:.1f}\\%" for p in below_threshold_t1_pct) + " \\\\\n\\hline\n"
    latex_str += "Below T2 & " + " & ".join(f"{b}" for b in below_threshold_t2) + " \\\\\n\\hline\n"
    latex_str += "< T2 Correct \\% & " + " & ".join(f"{p:.1f}\\%" for p in below_threshold_t2_pct) + " \\\\\n\\hline\n"
    latex_str += "Misclass Below T1 & " + " & ".join(f"{m}" for m in misclass_below_t1) + " \\\\\n\\hline\n"
    
    latex_str += "\\end{tabular}\n\\caption{Classification Results}\n\\end{table}"
    return latex_str

## Splitting logic and presentation

def compute_classification_statistics(train_data_with_distances, thresholds_same, thresholds_different):
    """
    Compute classification statistics for each digit.
    
    Parameters:
    train_data_with_distances: numpy array containing training data and distances
    thresholds_same: array of thresholds for same-digit classification
    thresholds_different: array of thresholds for different-digit classification
    
    Returns:
    dict: Dictionary containing various classification statistics
    """
    stats = {
        'totals': [],
        'incorrects': [],  # New statistic
        'corrects': [],
        'correct_percentages': [],
        'correct_above_t1': [],
        'correct_above_t1_pct': [],
        'correct_below_t1': [],
        'correct_below_t1_pct': [],
        'correct_below_t2': [],
        'correct_below_t2_pct': [],
        'misclass_below_t1': [],
        'misclass_below_t1_pct': [],
        'misclass_below_t2': []
    }
    
    for digit in range(10):
        # Get samples for this digit
        digit_mask = train_data_with_distances[:, 10] == digit
        digit_samples = train_data_with_distances[digit_mask]
        
        # Basic statistics
        total = len(digit_samples)
        correct_mask = digit_samples[:, 11] == digit_samples[:, 10]
        correct = np.sum(correct_mask)
        incorrect = total - correct  # New calculation
        correct_percentage = (correct / total) * 100
        
        # T1 threshold statistics
        below_t1_mask = digit_samples[:, 12] < thresholds_same[digit]
        correct_below_t1_mask = correct_mask & below_t1_mask
        correct_below_t1_count = np.sum(correct_below_t1_mask)
        correct_below_t1_percentage = (correct_below_t1_count / total) * 100
        
        # Correct above T1
        correct_above_t1_mask = correct_mask & ~below_t1_mask
        correct_above_t1_count = np.sum(correct_above_t1_mask)
        correct_above_t1_percentage = (correct_above_t1_count / total) * 100
        
        # T2 threshold statistics (correctly classified samples below T2)
        correct_samples = digit_samples[correct_mask]
        correct_below_t2_count = np.sum(correct_samples[:, 12] < thresholds_different[digit])
        correct_below_t2_percentage = (correct_below_t2_count / total) * 100
        
        # Misclassification statistics
        misclass_mask = (train_data_with_distances[:, 11] == digit) & \
                       (train_data_with_distances[:, 10] != digit)
        misclass_samples = train_data_with_distances[misclass_mask]
        misclass_below_t1_count = np.sum(misclass_samples[:, 13] < thresholds_same[digit])
        misclass_below_t1_percentage = (misclass_below_t1_count / total) * 100
        misclass_below_t2_count = np.sum(misclass_samples[:, 13] < thresholds_different[digit])
        
        # Store all statistics
        stats['totals'].append(total)
        stats['incorrects'].append(incorrect)  # New statistic
        stats['corrects'].append(correct)
        stats['correct_percentages'].append(correct_percentage)
        stats['correct_above_t1'].append(correct_above_t1_count)
        stats['correct_above_t1_pct'].append(correct_above_t1_percentage)
        stats['correct_below_t1'].append(correct_below_t1_count)
        stats['correct_below_t1_pct'].append(correct_below_t1_percentage)
        stats['correct_below_t2'].append(correct_below_t2_count)
        stats['correct_below_t2_pct'].append(correct_below_t2_percentage)
        stats['misclass_below_t1'].append(misclass_below_t1_count)
        stats['misclass_below_t1_pct'].append(misclass_below_t1_percentage)
        stats['misclass_below_t2'].append(misclass_below_t2_count)
        # Add total sums for key metrics
        stats['total_incorrect'] = sum(stats['incorrects'])
        stats['total_misclass_below_t1'] = sum(stats['misclass_below_t1'])        
    
    return stats

def generate_threshold_table(stats):
    """
    Generate a LaTeX table from the classification statistics.
    
    Parameters:
    stats: dict containing classification statistics
    
    Returns:
    str: LaTeX formatted table string
    """
    # Initialize the LaTeX table string
    latex_str = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|c|}\n\\hline\n"
    latex_str += "\\multicolumn{11}{|c|}{Thresholds} \\\\\n\\hline\n"
    latex_str += "Digit & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 \\\\\n\\hline\n"
    
    # Add rows in specified order with format specifications
    row_specs = [
        ('Total', 'totals', '{:d}'),
        ('Incorrect', 'incorrects', '{:d}'),  # New row
        ('Correct', 'corrects', '{:d}'),
        ('Correct \\%', 'correct_percentages', '{:.2f}\\%'),
        ('Correct Above T1', 'correct_above_t1', '{:d}'),
        ('Correct Above T1 \\%', 'correct_above_t1_pct', '{:.2f}\\%'),
        ('Correct Below T1', 'correct_below_t1', '{:d}'),
        ('Correct Below T1 \\%', 'correct_below_t1_pct', '{:.2f}\\%'),
        ('Correct Below T2', 'correct_below_t2', '{:d}'),
        ('Correct Below T2 \\%', 'correct_below_t2_pct', '{:.2f}\\%'),
        ('Misclass Below T1', 'misclass_below_t1', '{:d}'),
        ('Misclass Below T1 \\%', 'misclass_below_t1_pct', '{:.2f}\\%'),
        ('Misclass Below T2', 'misclass_below_t2', '{:d}')
    ]
    
    # Generate each row of the table
    for row_name, stat_key, format_str in row_specs:
        values = [format_str.format(val) for val in stats[stat_key]]
        latex_str += f"{row_name} & " + " & ".join(values) + " \\\\\n\\hline\n"
    
    # Add totals row at the end
    latex_str += "\\multicolumn{11}{|l|}{Total Incorrect: " + str(stats['total_incorrect']) + "} \\\\\n\\hline\n"
    latex_str += "\\multicolumn{11}{|l|}{Total Misclassifications Below T1: " + str(stats['total_misclass_below_t1']) + "} \\\\\n\\hline\n"
    
    latex_str += "\\end{tabular}\n\\caption{Classification Results}\n\\end{table}"
    return latex_str

import numpy as np
from scipy.spatial.distance import pdist

def compute_centroid_distances(centroids):
    """
    Compute statistics on pairwise Euclidean distances between centroids.
    
    Parameters:
    -----------
    centroids : numpy.ndarray
        Array of shape (n_centroids, dimensions) containing centroid vectors
    
    Returns:
    --------
    tuple
        (min_distance, max_distance, mean_distance) statistics on the pairwise distances
    """
    # Compute pairwise Euclidean distances
    pairwise_distances = pdist(centroids, metric='euclidean')
    
    # Calculate statistics
    min_distance = np.min(pairwise_distances)
    max_distance = np.max(pairwise_distances)
    mean_distance = np.mean(pairwise_distances)
    
    return min_distance, max_distance, mean_distance

# Example usage:
centroids = np.array([[9.92047086e-01, 2.80194969e-04, 1.27253789e-03, 9.16153109e-05,
                       2.76997533e-04, 6.65386512e-04, 3.59899877e-03, 4.54141829e-04,
                       4.89811152e-04, 8.23237843e-04],
                      [3.95979961e-04, 9.91423781e-01, 1.78550633e-03, 5.07866463e-04,
                       1.52409187e-03, 2.30262061e-04, 8.58913478e-04, 2.03972482e-03,
                       9.76951319e-04, 2.56925048e-04],
                      [2.06684753e-03, 4.10567246e-03, 9.80493523e-01, 4.04550732e-03,
                       1.17827286e-03, 1.93189402e-04, 5.19941896e-04, 4.75762701e-03,
                       2.20890443e-03, 4.30515882e-04],
                      [8.67938133e-04, 7.92726919e-04, 3.25205725e-03, 9.82288736e-01,
                       3.03486041e-05, 5.47235248e-03, 6.21448241e-05, 1.70498409e-03,
                       1.92571989e-03, 3.60299168e-03],
                      [6.41766637e-04, 1.19807115e-03, 6.25597911e-04, 3.28220380e-05,
                       9.85247046e-01, 9.24715020e-05, 2.06256431e-03, 2.39092107e-03,
                       5.04423310e-04, 7.20431736e-03],
                      [9.99830478e-04, 4.47734905e-04, 2.36704626e-04, 3.78398755e-03,
                       1.56336256e-04, 9.84799808e-01, 4.05731532e-03, 3.18621208e-04,
                       1.92523531e-03, 3.27442340e-03],
                      [2.18357458e-03, 6.67876119e-04, 2.55338914e-04, 6.56696169e-05,
                       9.85667608e-04, 2.35234386e-03, 9.92624007e-01, 3.12838794e-05,
                       7.76639605e-04, 5.76009516e-05],
                      [8.64703321e-04, 1.01745542e-03, 2.62277280e-03, 2.60410599e-03,
                       1.25067557e-03, 3.13882467e-04, 1.90881863e-05, 9.84164656e-01,
                       3.09810235e-04, 6.83284973e-03],
                      [3.22150087e-03, 2.86732538e-03, 4.58279337e-03, 5.17971236e-03,
                       1.43046210e-03, 4.34504898e-03, 4.87435983e-03, 1.09273969e-03,
                       9.65256154e-01, 7.14990621e-03],
                      [1.06570727e-03, 6.78994479e-04, 2.03841649e-04, 2.24301153e-03,
                       7.50650804e-03, 1.72871757e-03, 1.80209627e-04, 6.77323718e-03,
                       1.27662143e-03, 9.78343154e-01]])

def get_selected_rows(training_data_with_distances, indexes_same):
    """
    Select rows from training_data_with_distances based on indexes_same and return columns 0 to 9.
    
    Parameters:
    -----------
    training_data_with_distances : numpy.ndarray
        The array containing training data with distances.
    indexes_same : numpy.ndarray or list
        The vector of indexes to select specific rows.
    
    Returns:
    --------
    numpy.ndarray
        The selected rows with columns 0 to 9.
    """
    # Ensure indexes_same is a numpy array
    indexes_same = np.array(indexes_same, dtype=int)
    
    # Select the rows indexed by indexes_same and columns 0 to 9
    selected_rows = training_data_with_distances[indexes_same, :10]
    
    return selected_rows

from scipy.optimize import minimize
def fit_hypersphere_with_constraints(edges, centroids, d):
    # Ensure inputs are properly structured 2D arrays
    centroids = np.array(centroids, dtype=np.float64).reshape(10, 10)
    edges = np.array(edges, dtype=np.float64).reshape(10, 10)
    
    n_dim = centroids.shape[1]  # 10 dimensions
    
    def objective(params):
        c = params[:-1]
        r = params[-1]
        residuals = [np.linalg.norm(e - c) - r for e in edges]
        return np.sum(np.array(residuals)**2)
    
    # Constraints: ||e_i - m_i|| = d_i
    def constraint(params):
        c = params[:-1]
        edges_reconstructed = []
        for m_i, d_i in zip(centroids, d):
            delta = c - m_i
            norm = np.linalg.norm(delta)
            if norm < 1e-9:
                edge = m_i.copy()  # Avoid division by zero
            else:
                edge = m_i + (d_i / norm) * delta
            edges_reconstructed.append(edge)
        return np.concatenate(edges_reconstructed) - edges.flatten()
    
    # Initial guess: Use the centroid of edges as the starting center
    c_init = np.mean(edges, axis=0)
    r_init = np.mean([np.linalg.norm(e - c_init) for e in edges])
    initial_guess = np.concatenate([c_init, [r_init]])
    
    # Solve the constrained optimization
    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        constraints={'type': 'eq', 'fun': constraint},
        options={'maxiter': 1000, 'ftol': 1e-8}
    )
    
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    
    c_opt = result.x[:-1]
    r_opt = result.x[-1]
    return c_opt, r_opt

##########################
# MORE THRESHOLD STUDIES #
##########################

import numpy as np

def analyze_thresholds(train_data_with_distances, thresholds):
    """
    Analyze classification performance across different thresholds.
    
    Parameters:
    train_data_with_distances: numpy array containing the data
        - Columns 0-9: Softmax outputs
        - Column 10: True class
        - Column 11: Predicted class
        - Column 12: Distance to true class centroid
        - Column 13: Distance to predicted class centroid
    thresholds: list of threshold values to analyze
    
    Returns:
    dict: Statistics for each threshold containing counts and percentages
    """
    stats = {t: {
        'per_digit': [],  # Will store detailed stats for each digit
        'total_correct_within': 0,
        'total_incorrect_within': 0,
        'total_samples_within': 0,
        'accuracy_at_threshold': 0.0,
        'retention_rate': 0.0,  # Percentage of total samples retained at threshold
        'cumulative_correct_lost': 0,  # Running total of correct predictions lost
        'cumulative_incorrect_eliminated': 0  # Running total of incorrect predictions eliminated
    } for t in thresholds}
    
    total_samples = len(train_data_with_distances)
    
    for digit in range(10):
        # Get all samples where this is the true class
        true_digit_mask = train_data_with_distances[:, 10] == digit
        true_digit_samples = train_data_with_distances[true_digit_mask]
        
        # Split into correct and incorrect predictions
        correct_mask = true_digit_samples[:, 11] == digit
        correct_samples = true_digit_samples[correct_mask]
        incorrect_samples = true_digit_samples[~correct_mask]
        
        for t in thresholds:
            # For correct predictions, check distance to true centroid
            correct_within = np.sum(correct_samples[:, 12] < t)
            
            # For incorrect predictions, check distance to predicted centroid
            incorrect_within = np.sum(incorrect_samples[:, 13] < t)
            
            # Total samples within threshold for this digit
            total_within = correct_within + incorrect_within
            
            # Calculate accuracy for this digit at this threshold
            accuracy = correct_within / total_within if total_within > 0 else 0.0
            
            # Store per-digit statistics
            stats[t]['per_digit'].append({
                'digit': digit,
                'correct_within': correct_within,
                'incorrect_within': incorrect_within,
                'total_within': total_within,
                'accuracy': accuracy,
                'retention_rate': total_within / len(true_digit_samples) * 100
            })
            
            # Update totals
            stats[t]['total_correct_within'] += correct_within
            stats[t]['total_incorrect_within'] += incorrect_within
            stats[t]['total_samples_within'] += total_within
    
    # Calculate overall statistics for each threshold
    # Get baseline values from highest threshold
    max_threshold = max(thresholds)
    baseline_correct = stats[max_threshold]['total_correct_within']
    baseline_incorrect = stats[max_threshold]['total_incorrect_within']
    
    # Calculate cumulative losses/eliminations for each threshold
    for t in sorted(thresholds, reverse=True):
        stats[t]['cumulative_correct_lost'] = baseline_correct - stats[t]['total_correct_within']
        stats[t]['cumulative_incorrect_eliminated'] = baseline_incorrect - stats[t]['total_incorrect_within']
        
        total_within = stats[t]['total_samples_within']
        if total_within > 0:
            stats[t]['accuracy_at_threshold'] = (
                stats[t]['total_correct_within'] / total_within * 100
            )
            stats[t]['retention_rate'] = (
                total_within / total_samples * 100
            )

    return stats

def print_threshold_analysis(stats, thresholds):
    """
    Print a formatted analysis of the threshold statistics.
    
    Parameters:
    stats: dict containing the threshold statistics
    thresholds: list of threshold values
    """
    print("\nEnhanced Threshold Analysis")
    print("=" * 120)
    
    # Print header
    header = (
        f"{'Threshold':<10} {'Total Within':<12} {'Retention %':<12} "
        f"{'Accuracy %':<12} {'Correct':<10} {'Incorrect':<10} "
        f"{'Cum Lost':<10} {'Cum Elim':<10} {'Ratio':<10}"
    )
    print(header)
    print("-" * 120)
    
    # Print statistics for each threshold
    for t in thresholds:
        threshold_stats = stats[t]
        print(
            f"{t:<10.2f} "
            f"{threshold_stats['total_samples_within']:<12d} "
            f"{threshold_stats['retention_rate']:<12.2f} "
            f"{threshold_stats['accuracy_at_threshold']:<12.2f} "
            f"{threshold_stats['total_correct_within']:<10d} "
            f"{threshold_stats['total_incorrect_within']:<10d} "
            f"{threshold_stats['cumulative_correct_lost']:<10d} "
            f"{threshold_stats['cumulative_incorrect_eliminated']:<10d} "
            f"{int(round(threshold_stats['total_correct_within']/threshold_stats['total_incorrect_within'])):d}:1"
        )
    
    print("=" * 120)
    
    # Print per-digit breakdown
    print("\nPer-Digit Breakdown at Each Threshold")
    print("=" * 120)
    
    for t in thresholds:
        print(f"\nThreshold: {t:.2f}")
        print("-" * 120)
        print(f"{'Digit':<6} {'Within':<8} {'Retention %':<12} {'Accuracy %':<12}")
        
        for digit_stats in stats[t]['per_digit']:
            print(
                f"{digit_stats['digit']:<6d} "
                f"{digit_stats['total_within']:<8d} "
                f"{digit_stats['retention_rate']:<12.2f} "
                f"{digit_stats['accuracy']:<12.2f}"
            )

# https://claude.ai/chat/6cdf5543-4d2d-4968-8c45-2cebad2fb59a

import matplotlib.pyplot as plt
import numpy as np

def create_visualization(threshold_stats):
    # Extract thresholds and metrics
    thresholds = sorted(threshold_stats.keys(), reverse=True)  # Sort from highest to lowest
    
    # Extract overall metrics for each threshold
    metrics = {
        'total_correct': [],
        'total_incorrect': [],
        'retention': [],
        'accuracy': [],
        'cum_lost': [],
        'cum_elim': []
    }
    
    for t in thresholds:
        stats = threshold_stats[t]
        metrics['total_correct'].append(stats['total_correct_within'])
        metrics['total_incorrect'].append(stats['total_incorrect_within'])
        metrics['retention'].append(stats['retention_rate'])
        metrics['accuracy'].append(stats['accuracy_at_threshold'])
        metrics['cum_lost'].append(stats['cumulative_correct_lost'])
        metrics['cum_elim'].append(stats['cumulative_incorrect_eliminated'])

    # Create figure and subplots
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Core Trade-off (Cum Lost vs Incorrect)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(thresholds, metrics['cum_lost'], 'b.-', label='Cumulative Correct Lost')
    ax1.plot(thresholds, metrics['total_incorrect'], 'r.-', label='Incorrect Remaining')
    ax1.set_xlabel('Distance Threshold')
    ax1.set_ylabel('Number of Predictions')
    ax1.set_title('Trade-off: Correct Lost vs Incorrect Remaining')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Ratio Evolution (log scale)
    ratios = [correct/incorrect if incorrect > 0 else float('inf') 
              for correct, incorrect in zip(metrics['total_correct'], metrics['total_incorrect'])]
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(thresholds, ratios, 'g.-')
    ax2.set_xlabel('Distance Threshold')
    ax2.set_ylabel('Ratio (Correct:Incorrect)')
    ax2.set_title('Evolution of Correct:Incorrect Ratio')
    ax2.grid(True)

    # Plot 3: Retention vs Accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(thresholds, metrics['retention'], 'b.-', label='Retention %')
    ax3.plot(thresholds, metrics['accuracy'], 'r.-', label='Accuracy %')
    ax3.set_xlabel('Distance Threshold')
    ax3.set_ylabel('Percentage')
    ax3.set_title('Retention and Accuracy Trade-off')
    ax3.legend()
    ax3.grid(True)

    # Plot 4: Per-digit Retention Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Get per-digit retention for first and last threshold
    first_threshold = thresholds[0]
    last_threshold = thresholds[-1]
    
    digits = range(10)
    retention_first = [threshold_stats[first_threshold]['per_digit'][d]['retention_rate'] 
                      for d in digits]
    retention_last = [threshold_stats[last_threshold]['per_digit'][d]['retention_rate'] 
                     for d in digits]
    
    x = np.arange(10)
    width = 0.35
    
    ax4.bar(x - width/2, retention_first, width, label=f'Threshold {first_threshold}')
    ax4.bar(x + width/2, retention_last, width, label=f'Threshold {last_threshold}')
    
    ax4.set_xlabel('Digit')
    ax4.set_ylabel('Retention Rate (%)')
    ax4.set_title('Per-digit Retention Rate Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(digits)
    ax4.legend()
    ax4.grid(True)

    # Add overall title
    fig.suptitle('MNIST Classification Performance Analysis\nThreshold-based Distance to Centroids', 
                 fontsize=16, y=0.95)

    # Adjust layout
    plt.tight_layout()
    return fig

# Usage:
# fig = create_visualization(threshold_stats)
# plt.show()            

###########################
# CENTROID DISTANCE STATS #
###########################

def get_num_pairs(n):
    """
    Calculate number of unique pairs from n elements.
    
    Parameters:
    -----------
    n : int
        Number of elements
    
    Returns:
    --------
    int
        Number of unique pairs possible from n elements
    """
    return (n * (n-1)) // 2

get_num_pairs(10) # 45

import numpy as np
from itertools import combinations

def compute_pairwise_distances(centroids):
    """
    Compute all pairwise Euclidean distances between centroids.
    
    Parameters:
    -----------
    centroids : numpy.ndarray
        Array of shape (n_centroids, dimensions) containing centroid vectors
    
    Returns:
    --------
    numpy.ndarray
        Array of pairwise distances between all unique centroid pairs
    """
    distances = []
    for i, j in combinations(range(len(centroids)), 2):
        dist = np.sqrt(np.sum((centroids[i] - centroids[j])**2))
        distances.append(dist)
    return np.array(distances)


import numpy as np

def compute_meta_centroid_arithmetic(centroids):
    """
    Compute the arithmetic mean of all centroids (simple average).
    
    Parameters:
    -----------
    centroids : numpy.ndarray
        Array of shape (n_centroids, dimensions) containing centroid vectors
    
    Returns:
    --------
    numpy.ndarray
        The arithmetic meta-centroid vector
    """
    return np.mean(centroids, axis=0)

def compute_meta_centroid_weighted(centroids):
    """
    Compute a weighted average of centroids, with weights inversely proportional
    to their average distance from other centroids.
    
    Parameters:
    -----------
    centroids : numpy.ndarray
        Array of shape (n_centroids, dimensions) containing centroid vectors
    
    Returns:
    --------
    numpy.ndarray
        The weighted meta-centroid vector
    """
    n_centroids = len(centroids)
    distances = np.zeros((n_centroids, n_centroids))
    
    for i in range(n_centroids):
        for j in range(n_centroids):
            distances[i,j] = np.sqrt(np.sum((centroids[i] - centroids[j])**2))
    
    avg_distances = np.mean(distances, axis=1)
    weights = 1 / (avg_distances + np.finfo(float).eps)
    weights = weights / np.sum(weights)
    
    return np.average(centroids, weights=weights, axis=0)

def compute_meta_centroid_geometric(centroids):
    """
    Compute the geometric mean of all centroids. Works by taking the 
    element-wise logarithm, averaging, then exponentiating back.
    
    Parameters:
    -----------
    centroids : numpy.ndarray
        Array of shape (n_centroids, dimensions) containing centroid vectors
    
    Returns:
    --------
    numpy.ndarray
        The geometric meta-centroid vector, normalized to sum to 1
    """
    log_centroids = np.log(centroids + np.finfo(float).eps)
    avg_log = np.mean(log_centroids, axis=0)
    unnormalized = np.exp(avg_log)
    return unnormalized / np.sum(unnormalized)


def compute_meta_centroid_medoid(centroids):
    """
    Find the medoid centroid, which is the centroid with minimum total
    distance to all other centroids.
    
    Parameters:
    -----------
    centroids : numpy.ndarray
        Array of shape (n_centroids, dimensions) containing centroid vectors
    
    Returns:
    --------
    tuple
        (medoid_centroid, medoid_index) - The medoid centroid vector and its index
    """
    n_centroids = len(centroids)
    distances = np.zeros((n_centroids, n_centroids))
    
    # Compute pairwise distances
    for i in range(n_centroids):
        for j in range(n_centroids):
            distances[i,j] = np.sqrt(np.sum((centroids[i] - centroids[j])**2))
    
    # Sum of distances for each centroid to all others
    total_distances = np.sum(distances, axis=1)
    
    # Find centroid with minimum total distance
    medoid_idx = np.argmin(total_distances)
    
    return centroids[medoid_idx], medoid_idx


# medoid, medoid_idx = compute_meta_centroid_medoid(centroids)
# print(f"\nMedoid centroid (index {medoid_idx}):")
# print(medoid)
# print(f"Sum: {np.sum(medoid):.6f}")
# print(f"Average distance to original centroids: {np.mean([np.sqrt(np.sum((c - medoid)**2)) for c in centroids]):.6f}")

## store results

def compute_avg_distance(centroid, centroids):
    """
    Compute the average Euclidean distance from a centroid to all centroids in a set.
    
    Parameters:
    -----------
    centroid : numpy.ndarray
        The reference centroid vector
    centroids : numpy.ndarray
        Array of shape (n_centroids, dimensions) containing centroid vectors
    
    Returns:
    --------
    float
        Average Euclidean distance from the reference centroid to all centroids
    """
    return np.mean([np.sqrt(np.sum((c - centroid)**2)) for c in centroids])

# meta_centroids = {
#     'arithmetic': {
#         'values': meta_arithmetic,
#         'avg_distance': compute_avg_distance(meta_arithmetic, centroids)
#     },
#     'weighted': {
#         'values': meta_weighted,
#         'avg_distance': compute_avg_distance(meta_weighted, centroids)
#     },
#     'geometric': {
#         'values': meta_geometric,
#         'avg_distance': compute_avg_distance(meta_geometric, centroids)
#     }
# }

# https://claude.ai/chat/92e05802-23d9-4d39-8da5-e1d5e68e329a

def add_meta_centroid_distances(train_data_with_distances, meta_centroid):
   """
   Compute distances between each prediction's softmax output and the meta centroid.
   
   Parameters:
   -----------
   train_data_with_distances : numpy.ndarray
       Array containing softmax outputs and metadata in columns
   meta_centroid : numpy.ndarray
       The geometric meta centroid vector
       
   Returns:
   --------
   numpy.ndarray
       Updated array with meta centroid distances in column 14
   """
   # Create copy of input array
   updated_data = np.copy(train_data_with_distances)
   
   # Extract softmax outputs (columns 0-9)
   softmax_outputs = updated_data[:, :10]
   
   # Compute Euclidean distances to meta centroid
   meta_distances = np.sqrt(np.sum((softmax_outputs - meta_centroid)**2, axis=1))
   
   # Add new column with meta centroid distances
   updated_data = np.column_stack([updated_data, meta_distances])
   
   return updated_data

import numpy as np

def populate_cluster_stats(train_data_with_meta, centroids, meta_centroid_avg_dist):
    # Initialize structure
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    N = len(train_data_with_meta)
    
    cluster_stats = {
        'thresholds': thresholds,
        'class_clusters': {
            digit: {
                'correct': [],
                'incorrect': []
            } for digit in range(10)
        },
        'meta_cluster': {'members': []},
        'no_membership': [],
        'total_examples': N
    }
    
    # For each threshold
    for threshold in thresholds:
        # Initialize counters for this threshold
        class_members = {d: {'correct': 0, 'incorrect': 0} for d in range(10)}
        meta_members = 0
        
        # Calculate meta-centroid radius for this threshold
        meta_radius = meta_centroid_avg_dist - threshold
        
        # Check membership for each example
        for example in train_data_with_meta:
            softmax = example[0:10]
            true_class = int(example[10])
            pred_class = int(example[11])
            true_centroid_dist = example[12]
            meta_centroid_dist = example[14]
            
            # Check class cluster memberships
            for digit in range(10):
                distance = np.sqrt(np.sum((softmax - centroids[digit])**2))
                if distance <= threshold:
                    if digit == true_class:
                        class_members[digit]['correct'] += 1
                    else:
                        class_members[digit]['incorrect'] += 1
            
            # Check meta-centroid membership
            if meta_centroid_dist <= meta_radius:
                meta_members += 1
        
        # Store results for this threshold
        for digit in range(10):
            cluster_stats['class_clusters'][digit]['correct'].append(
                class_members[digit]['correct'])
            cluster_stats['class_clusters'][digit]['incorrect'].append(
                class_members[digit]['incorrect'])
            
        cluster_stats['meta_cluster']['members'].append(meta_members)
        
        # Calculate points with no membership
        total_in_clusters = sum(
            class_members[d]['correct'] + class_members[d]['incorrect'] 
            for d in range(10)
        ) + meta_members
        
        no_membership = N - total_in_clusters
        cluster_stats['no_membership'].append(no_membership)
    
    return cluster_stats

def validate_cluster_stats(cluster_stats, train_data_with_meta):
    N = len(train_data_with_meta)
    validations = []
    
    # 1. Basic count validations
    for t_idx, threshold in enumerate(cluster_stats['thresholds']):
        # Total examples should equal N
        total_correct = sum(cluster_stats['class_clusters'][d]['correct'][t_idx] 
                          for d in range(10))
        total_incorrect = sum(cluster_stats['class_clusters'][d]['incorrect'][t_idx] 
                            for d in range(10))
        meta_members = cluster_stats['meta_cluster']['members'][t_idx]
        no_members = cluster_stats['no_membership'][t_idx]
        
        validations.append({
            'threshold': threshold,
            'total_accounted_for': (total_correct + total_incorrect + 
                                  meta_members + no_members == N),
            'counts_non_negative': all(
                cluster_stats['class_clusters'][d]['correct'][t_idx] >= 0 and
                cluster_stats['class_clusters'][d]['incorrect'][t_idx] >= 0
                for d in range(10)
            )
        })
        
    # 2. Monotonicity check - counts should generally increase as threshold increases
    for digit in range(10):
        correct_counts = cluster_stats['class_clusters'][digit]['correct']
        incorrect_counts = cluster_stats['class_clusters'][digit]['incorrect']
        
        validations.append({
            'digit': digit,
            'correct_monotonic': all(correct_counts[i] <= correct_counts[i+1] 
                                   for i in range(len(correct_counts)-1)),
            'incorrect_monotonic': all(incorrect_counts[i] <= incorrect_counts[i+1] 
                                     for i in range(len(incorrect_counts)-1))
        })
    
    # 3. Meta-cluster validation
    meta_counts = cluster_stats['meta_cluster']['members']
    validations.append({
        'meta_monotonic': all(meta_counts[i] <= meta_counts[i+1] 
                             for i in range(len(meta_counts)-1)),
        'meta_non_negative': all(count >= 0 for count in meta_counts)
    })
    
    # 4. Cross-validation with actual data
    correct_predictions = sum(1 for example in train_data_with_meta 
                            if example[10] == example[11])
    total_correct_in_clusters = sum(
        cluster_stats['class_clusters'][d]['correct'][0]  # at highest threshold
        for d in range(10)
    )
    validations.append({
        'correct_predictions_match': correct_predictions >= total_correct_in_clusters
    })
    
    return validations

def print_validation_results(validations):
    print("Validation Results:")
    print("-" * 50)
    
    # Print threshold-specific validations
    for v in validations:
        if 'threshold' in v:
            print(f"\nThreshold {v['threshold']}:")
            print(f"Total examples accounted for: {v['total_accounted_for']}")
            print(f"All counts non-negative: {v['counts_non_negative']}")
        
        elif 'digit' in v:
            print(f"\nDigit {v['digit']} monotonicity:")
            print(f"Correct counts monotonic: {v['correct_monotonic']}")
            print(f"Incorrect counts monotonic: {v['incorrect_monotonic']}")
        
        elif 'meta_monotonic' in v:
            print("\nMeta-cluster validations:")
            print(f"Monotonicity: {v['meta_monotonic']}")
            print(f"Non-negative counts: {v['meta_non_negative']}")
        
        elif 'correct_predictions_match' in v:
            print("\nCross-validation:")
            print(f"Correct predictions consistency: {v['correct_predictions_match']}")
            
# Claude 3.7 solution - written in Claude code            
def create_alphabetic_prediction_threshold_table(data_np):
    """
    Creates a table showing alphabetic character predictions in relation to distance thresholds.
    
    Parameters:
    -----------
    data_np : numpy.ndarray
        Array containing:
        - Columns 0-9: Softmax outputs
        - Column 10: True class
        - Column 11: Predicted class
        - Column 12: Distance to true class centroid
        - Column 13: Distance to predicted class centroid
        
    Returns:
    --------
    numpy.ndarray
        Array with rows representing different thresholds and columns representing:
        - Column 0: Threshold values (0.9 to 0.1 in decrements of 0.1, and 0.05)
        - Columns 1-10: Number of alphabetic chars predicted as digits 0-9 below threshold
        - Column 11: TBT (Total below threshold)
        - Column 12: TAT (Total above threshold)
    """
    # Define thresholds
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    
    # Initialize result array
    # Columns: Threshold, Digits 0-9, TBT, TAT
    result = np.zeros((len(thresholds), 13))
    
    # Filter to only include alphabetic characters (true labels 10-61)
    alpha_mask = (data_np[:, 10] >= 10) & (data_np[:, 10] <= 61)
    alpha_data = data_np[alpha_mask]
    
    # Total number of alphabetic characters
    total_alpha = len(alpha_data)
    
    # For each threshold
    for i, threshold in enumerate(thresholds):
        # Set threshold value in first column
        result[i, 0] = threshold
        
        # Track total below threshold across all digits
        total_below_threshold = 0
        
        # For each digit (0-9)
        for digit in range(10):
            # Find alphabetic characters predicted as this digit
            digit_mask = (alpha_data[:, 11] == digit)
            alpha_pred_as_digit = alpha_data[digit_mask]
            
            # Count those below the threshold distance
            # We use column 13 (distance to predicted class centroid)
            below_threshold = np.sum(alpha_pred_as_digit[:, 13] < threshold)
            
            # Store in appropriate column (digit + 1 to account for threshold column)
            result[i, digit + 1] = below_threshold
            
            # Add to total below threshold
            total_below_threshold += below_threshold
        
        # Set TBT (Total Below Threshold)
        result[i, 11] = total_below_threshold
        
        # Set TAT (Total Above Threshold)
        result[i, 12] = total_alpha - total_below_threshold
    
    return result


# https://chat.deepseek.com/a/chat/s/c0a4ad31-92ae-41cb-93cf-15f3530b71b9
def generate_table(data_np):
    # Filter rows where the true class is alphabetic (>=10)
    alphabetic_mask = data_np[:, 10] >= 10
    alphabetic_data = data_np[alphabetic_mask]
    
    if alphabetic_data.size == 0:
        return np.empty((0, 13))  # Return empty array if no data
    
    # Extract predicted classes and distances
    predicted_classes = alphabetic_data[:, 11].astype(int)
    distances = alphabetic_data[:, 13]
    total_samples = len(alphabetic_data)
    
    # Define thresholds
    thresholds = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    num_thresholds = len(thresholds)
    
    # Vectorized computation of distance masks
    distance_masks = distances[:, np.newaxis] <= thresholds
    
    # Initialize counts for each digit and threshold
    counts = np.zeros((10, num_thresholds), dtype=int)
    for d in range(10):
        mask_d = (predicted_classes == d)
        counts[d] = np.sum(mask_d[:, np.newaxis] & distance_masks, axis=0)
    
    # Compute TBT and TAT
    tbt = counts.sum(axis=0)
    tat = total_samples - tbt
    
    # Build the result array
    result = []
    for i in range(num_thresholds):
        th = thresholds[i]
        row = [th] + counts[:, i].tolist() + [tbt[i], tat[i]]
        result.append(row)
    
    return np.array(result)

# ChatGPT 03-mini-high
# https://chatgpt.com/c/67c44e41-82b4-800f-9352-9ff02e186a5b
def create_alphabetic_table(data_np):
    """
    Creates a summary table for alphabetic characters (true class >= 10)
    from the provided data_np array.

    The input data_np is assumed to have the following columns:
      - Columns 0-9: Softmax outputs (for digits 0-9)
      - Column 10: True class (0-9 for digits, >=10 for alphabetic characters)
      - Column 11: Predicted class (always a digit 0-9)
      - Column 12: Distance to true class centroid
      - Column 13: Distance to predicted class centroid

    The output table has 13 columns:
      - Column 0: "Threshold" with values [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02
      - Columns 1-10: Counts for each predicted digit (0-9) where the distance (column 13) is <= threshold.
      - Column 11: "TBT" (Total Below Threshold) = total rows with distance <= threshold.
      - Column 12: "TAT" (Total Above Threshold) = total rows with distance > threshold.
    """
    # Define the thresholds (in order: 0.9, 0.8, ..., 0.1, then 0.05)
    thresholds = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    
    # Filter to only include rows for alphabetic characters.
    # We assume alphabetic characters have true class >= 10.
    alpha_mask = data_np[:, 10] >= 10
    alpha_data = data_np[alpha_mask]
    
    # For alphabetic rows, use the predicted class (column 11) and
    # the distance to the predicted class centroid (column 13).
    predicted_digits = alpha_data[:, 11].astype(int)  # Ensure integer type for digit matching.
    distances = alpha_data[:, 13]
    
    table_rows = []
    
    # Loop over each threshold value.
    for thresh in thresholds:
        row = [thresh]
        # For each predicted digit (0 to 9), count rows with distance <= thresh.
        digit_counts = [
            np.sum((predicted_digits == d) & (distances <= thresh))
            for d in range(10)
        ]
        row.extend(digit_counts)
        # TBT: Total Below Threshold (all rows with distance <= thresh).
        tbt = np.sum(distances <= thresh)
        # TAT: Total Above Threshold (all rows with distance > thresh).
        tat = np.sum(distances > thresh)
        row.extend([tbt, tat])
        table_rows.append(row)
    
    # Convert to numpy array and return.
    return np.array(table_rows)

# Example usage:
# table = create_alphabetic_table(data_np)
# print(table)

# Grok 3 - modded for range 0.8 to 0.02
# https://x.com/i/grok?conversation=1896175820466475142
def create_alpha_threshold_table(data_np):
    # Define thresholds from 0.8 to 0.1 in steps of 0.1, plus 0.05 and 0.02
    thresholds = np.concatenate([np.arange(0.8, 0.0, -0.1), [0.05], [0.02]])
    
    # Identify alphabetic characters (true class >= 10 since 0-9 are digits)
    alpha_mask = data_np[:, 10] >= 10  # Column 10 is true class
    alpha_data = data_np[alpha_mask]
    
    # Initialize output array: 10 rows (thresholds) × 13 columns
    # Columns: Threshold, 0-9, TBT, TAT
    table_data = np.zeros((len(thresholds), 13))
    
    # Set threshold values in first column
    table_data[:, 0] = thresholds
    
    # For each threshold
    for i, thresh in enumerate(thresholds):
        # Get rows where predicted class is 0-9 (digits) for alphabetic chars
        pred_digits = alpha_data[:, 11]  # Column 11 is predicted class
        distances = alpha_data[:, 13]    # Column 13 is distance to predicted centroid
        
        # Count predictions for each digit (0-9) below threshold
        for digit in range(10):
            digit_mask = (pred_digits == digit) & (distances <= thresh) & (~np.isnan(distances))
            table_data[i, digit + 1] = np.sum(digit_mask)
        
        # Calculate TBT (Total Below Threshold)
        below_mask = (distances <= thresh) & (~np.isnan(distances))
        table_data[i, 11] = np.sum(below_mask)
        
        # Calculate TAT (Total Above Threshold)
        above_mask = (distances > thresh) & (~np.isnan(distances))
        table_data[i, 12] = np.sum(above_mask)
    
    return table_data

# Example usage:
# result = create_alpha_threshold_table(data_np)
# For your specific example, result would be a 10×13 array where:
# - Column 0: [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
# - Columns 1-10: Counts of alphabetic chars predicted as digits 0-9 below threshold
# - Column 11: TBT (sum of columns 1-10 for that threshold)
# - Column 12: TAT (count of predictions above threshold)

# import os
# import numpy as np
# import matplotlib.pyplot as plt

# https://chatgpt.com/c/67c46b79-8f2c-800f-a137-5b64194514e0
def create_and_save_heatmaps(data_np, filename=None, save=False):
    """
    Creates two side-by-side heatmaps for uppercase (A-Z) and lowercase (a-z)
    misclassifications as digits (0-9) from the provided data_np and saves the
    figure to ../figures directory.
    
    Parameters:
        data_np (np.array): The numpy array containing the prediction data.
        save_filename (str): The filename for the saved figure.
    """
    # Extract true and predicted classes
    true_classes = data_np[:, 10]
    pred_classes = data_np[:, 11]

    # Create counts for uppercase letters (A-Z, true classes 10-35)
    upper_counts = np.zeros((26, 10), dtype=int)
    mask_upper = (true_classes >= 10) & (true_classes <= 35) & (pred_classes < 10)
    for t, p in zip(true_classes[mask_upper], pred_classes[mask_upper]):
        row_idx = int(t) - 10  # Map true class 10->row 0, ..., 35->row 25
        col_idx = int(p)       # Digit 0-9
        upper_counts[row_idx, col_idx] += 1

    # Create counts for lowercase letters (a-z, true classes 36-61)
    lower_counts = np.zeros((26, 10), dtype=int)
    mask_lower = (true_classes >= 36) & (true_classes <= 61) & (pred_classes < 10)
    for t, p in zip(true_classes[mask_lower], pred_classes[mask_lower]):
        row_idx = int(t) - 36  # Map true class 36->row 0, ..., 61->row 25
        col_idx = int(p)
        lower_counts[row_idx, col_idx] += 1

    # Create figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))

    # Fixed color range from 0 to 60 for consistency in color mapping
    vmin, vmax = 0, 60

    # Plot uppercase heatmap (without a colorbar)
    im0 = axs[0].imshow(upper_counts, cmap="Reds", aspect="auto", vmin=vmin, vmax=vmax)
    # Annotate the counts for uppercase
    for i in range(upper_counts.shape[0]):
        for j in range(upper_counts.shape[1]):
            axs[0].text(j, i, str(upper_counts[i, j]), ha="center", va="center", color="black")
    axs[0].set_xticks(np.arange(10))
    axs[0].set_xticklabels([str(d) for d in range(10)])
    axs[0].set_yticks(np.arange(26))
    axs[0].set_yticklabels([chr(65 + i) for i in range(26)])  # 'A' to 'Z'
    axs[0].set_title("Uppercase Misclassifications")

    # Plot lowercase heatmap (with a colorbar)
    im1 = axs[1].imshow(lower_counts, cmap="Reds", aspect="auto", vmin=vmin, vmax=vmax)
    # Annotate the counts for lowercase
    for i in range(lower_counts.shape[0]):
        for j in range(lower_counts.shape[1]):
            axs[1].text(j, i, str(lower_counts[i, j]), ha="center", va="center", color="black")
    axs[1].set_xticks(np.arange(10))
    axs[1].set_xticklabels([str(d) for d in range(10)])
    axs[1].set_yticks(np.arange(26))
    axs[1].set_yticklabels([chr(97 + i) for i in range(26)])  # 'a' to 'z'
    axs[1].set_title("Lowercase Misclassifications")

    # Add a colorbar only to the second heatmap
    cbar = fig.colorbar(im1, ax=axs[1])
    cbar.set_label("Count", rotation=270, labelpad=15)

    plt.tight_layout()

    if save and filename:
        # Ensure the save directory exists
        # save_dir = os.path.join("..", "figures")
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, save_filename)

        # # Save the figure
        # plt.savefig(save_path)
        # plt.close(fig)
        # print(f"Figure saved to {save_path}")

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved alphabetic misclassifications {filename}")          

# Example usage:
# Assuming data_np is already defined:
# create_and_save_heatmaps(data_np)
