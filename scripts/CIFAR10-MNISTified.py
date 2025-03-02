import numpy as np
import torch
from PIL import Image
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt
from datasets import load_dataset
from CIFAR10_helper_functions import convert_cifar10_to_mnist_format

def predict_cifar10_dataset(dataset, model, transform, device='cpu'):
    """
    Processes a Hugging Face CIFAR10 dataset split by converting each image to 
    MNIST-like format and predicting its class using the provided model.
    
    Args:
        dataset: A Hugging Face dataset split (e.g., test_ds).
        model: Trained PyTorch model.
        transform: Transform to convert PIL images to tensor and normalize.
        device (str): Device for computation ('cpu' or 'cuda').
    
    Returns:
        numpy.ndarray: Array of shape (N, 12) with softmax probabilities (10),
                       true label, and predicted label.
    """
    data_np = np.empty((0, 12))
    total = 0
    correct = 0
    
    for idx, item in enumerate(dataset):
        # Get the CIFAR10 image (PIL Image or convert from numpy)
        image = item['img']
        image_array = np.array(image)  # Convert PIL image to numpy array
        
        # Convert to MNIST-like format: 28x28 grayscale.
        mnist_img = convert_cifar10_to_mnist_format(image_array)
        pil_img = Image.fromarray(mnist_img)
        
        # Prepare tensor input.
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output_log_probs = model(input_tensor)
            # Convert log probabilities to probabilities.
            output_probs = torch.exp(output_log_probs)
            _, predicted = torch.max(output_probs.data, 1)
            predicted_label = predicted.item()
        
        # True label from the dataset.
        true_label = item['label']
        total += 1
        if predicted_label == true_label:
            correct += 1
        
        # Gather output: 10 probabilities, true label, and predicted label.
        output_np = output_probs.cpu().numpy()[0]
        row = np.hstack((output_np, np.array([true_label, predicted_label])))
        data_np = np.vstack((data_np, row))
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} images")
    
    print(f"Finished processing {total} images")
    print(f"Accuracy: {correct / total:.2%}")
    return data_np

def save_prediction_data(data_np, output_path):
    """
    Saves the prediction data (probabilities, true labels, and predicted labels) to a CSV file.
    """
    columns = [f'prob_{i}' for i in range(10)] + ['true_label', 'predicted_label']
    df = pd.DataFrame(data_np, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"Prediction data saved to {output_path}")
    return df

def main(dataset_split, model_path, output_path, device='cpu'):
    """
    Main function to run the prediction pipeline on a Hugging Face CIFAR10 dataset split.
    
    Args:
        dataset_split: A Hugging Face dataset split (e.g., test_ds).
        model_path (str): Path to the trained model file.
        output_path (str): CSV file path to save predictions.
        device (str): Device for computation ('cpu' or 'cuda').
    
    Returns:
        dict: Contains the NumPy prediction array and the Pandas DataFrame of results.
    """
    # Define the transformation: convert to tensor and normalize.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Define the CNN model architecture (must match the saved model).
    import torch.nn as nn
    import torch.nn.functional as F
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(-1, 32 * 7 * 7)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    # Load the model.
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("Predicting MNISTified CIFAR10 images from Hugging Face dataset...")
    data_np = predict_cifar10_dataset(dataset_split, model, transform, device)
    df = save_prediction_data(data_np, output_path)
    
    return {'data_np': data_np, 'dataframe': df}

# === Example usage ===

# Load CIFAR10 using Hugging Face datasets.
train_ds, test_ds = load_dataset('cifar10', split=['train[:50000]', 'test[:10000]'])

# Run prediction on the test set.
results = main(
    dataset_split=train_ds, 
    model_path='/home/daniel/git/work-in-progress/scripts/models/mnist_vanilla_cnn_local_202306241859.pth',      # Replace with your model file path
    output_path='cifar10_predictions.csv',
    device='cpu'  # or 'cuda'
)

data_np = results['data_np']
df = results['dataframe']

# Save the prediction NumPy array for further analysis.
np.save('data/cifar10_50k_mnistified_predictions.npy', data_np)
print("NumPy array saved as cifar10_50k_mnistified_predictions.npy")
