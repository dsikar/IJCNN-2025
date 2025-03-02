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