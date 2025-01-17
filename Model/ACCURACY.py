import torch
import torch.nn as nn
from torchvision.transforms import functional as F
from PIL import Image
import os
import numpy as np
import random
import matplotlib.pyplot as plt

# Define function to calculate accuracy
def calculate_accuracy(predicted, ground_truth):
    """
    Calculate accuracy as the percentage of correctly predicted pixels.
    """
    predicted = (predicted > 0.5).int()  # Convert to binary mask
    ground_truth = (ground_truth > 0.5).int()  # Ensure ground truth is binary
    correct = (predicted == ground_truth).sum().item()
    total = ground_truth.numel()
    accuracy = correct / total
    return accuracy

def test_model(model, test_folder, device, mask_folder, percentage=20, random_state=21):
    """
    Test the model on a random subset of images (percentage) from the test folder.
    Args:
        model: Trained model.
        test_folder: Path to folder containing test images.
        device: Torch device (e.g., 'cuda' or 'cpu').
        mask_folder: Path to folder containing ground truth masks.
        percentage: Percentage of test images to use for accuracy calculation.
        random_state: Fixed random seed for reproducibility.
    """
    model.to(device)  # Ensure the model is on the correct device
    model.eval()  # Set the model to evaluation mode
    
    # Set the random seed for reproducibility
    random.seed(random_state)  # Python's random seed
    np.random.seed(random_state)  # For NumPy randomness (if used)
    torch.manual_seed(random_state)  # PyTorch's random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)  # For CUDA

    accuracies = []
    test_images = os.listdir(test_folder)
    mask_images = os.listdir(mask_folder)

    if len(test_images) != len(mask_images):
        raise ValueError("Mismatch: Number of test images and mask images must be equal.")

    # Select 20% of the images randomly
    num_images_to_test = int(len(test_images) * percentage / 100)
    selected_images = random.sample(test_images, num_images_to_test)

    with torch.no_grad():  # No gradient computation
        for test_img, mask_img in zip(selected_images, mask_images):
            try:
                # Load the test image
                img_path = os.path.join(test_folder, test_img)
                image = Image.open(img_path).convert("RGB")
                image = image.resize((256, 256))  # Resize to a smaller size
                input_tensor = F.to_tensor(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

                # Load the ground-truth mask
                mask_path = os.path.join(mask_folder, mask_img)
                ground_truth = Image.open(mask_path).convert("L")
                ground_truth = ground_truth.resize((256, 256))  # Resize to match the input size
                ground_truth_tensor = F.to_tensor(ground_truth).to(device)  # Move mask to the same device

                # Get model prediction
                predicted_mask = model(input_tensor)

                # Resize prediction to ground truth size if needed
                if predicted_mask.size()[2:] != ground_truth_tensor.size()[1:]:
                    predicted_mask = F.resize(predicted_mask, ground_truth_tensor.size()[1:])

                # Calculate accuracy for the current image
                accuracy = calculate_accuracy(predicted_mask.squeeze(), ground_truth_tensor.squeeze())
                accuracies.append(accuracy)

            except Exception as e:
                print(f"Error processing image {test_img}: {e}")

    # Handle empty accuracy list to avoid NaN
    if accuracies:
        avg_accuracy = np.mean(accuracies)
        return avg_accuracy
    else:
        return 0.0

# Define your model architecture
class LaneDetectionCNN(nn.Module):
    def __init__(self):
        super(LaneDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)  # Define pooling layer
        self.fc1 = None  # Initialize later dynamically
        self.fc2 = nn.Linear(256, 64 * 64)  # No change here

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the feature map

        if not self.fc1:  # Initialize fc1 dynamically
            self.fc1 = nn.Linear(x.size(1), 256).to(x.device)

        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).view(-1, 1, 64, 64)

# Load the model from the .pth file
model_path = "C://lane_detection//model//cnn_lane_model.pth"  # Replace with your .pth file path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = LaneDetectionCNN()  # Initialize your model
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load weights
    model.to(device)
except Exception as e:
    print(f"Error loading model: {e}")

# Set paths and test the model
test_folder = "C://lane_detection//training_images//input"  # Path to the folder containing 20% test images
mask_folder = "C://lane_detection//training_images//output"  # Path to the folder containing ground-truth masks

try:
    test_accuracy = test_model(model, test_folder, device, mask_folder)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
except Exception as e:
    print(f"Error during testing: {e}")

# Process and visualize predictions for just two specific images
def test_two_images(model, test_folder, mask_folder, device):
    """
    Test the model on two specific images and visualize predictions.
    Args:
        model: Trained model.
        test_folder: Path to folder containing test images.
        mask_folder: Path to folder containing ground truth masks.
        device: Torch device (e.g., 'cuda' or 'cpu').
    """
    model.eval()  # Set the model to evaluation mode
    
    # Select two specific test images
    test_images = os.listdir(test_folder)[:2]  # Choose first two images
    mask_images = os.listdir(mask_folder)[:2]  # Choose corresponding masks
    
    fig, axes = plt.subplots(len(test_images), 3, figsize=(15, 10))
    
    with torch.no_grad():
        for i, (test_img, mask_img) in enumerate(zip(test_images, mask_images)):
            # Load the test image
            img_path = os.path.join(test_folder, test_img)
            mask_path = os.path.join(mask_folder, mask_img)
            
            # Check if paths exist
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"Error: {test_img} or its mask not found!")
                continue
            
            # Open and preprocess images
            image = Image.open(img_path).convert("RGB")
            image = image.resize((256, 256))  # Resize to a smaller size
            input_tensor = F.to_tensor(image).unsqueeze(0).to(device)  # Add batch dimension
            
            ground_truth = Image.open(mask_path).convert("L")
            ground_truth = ground_truth.resize((256, 256))  # Resize to match the input size
            ground_truth_tensor = F.to_tensor(ground_truth).to(device)  # Convert to tensor
            
            # Get prediction from the model
            predicted_mask = model(input_tensor)
            
            # Resize prediction to ground truth size if needed
            if predicted_mask.size()[2:] != ground_truth_tensor.size()[1:]:
                predicted_mask = F.resize(predicted_mask, ground_truth_tensor.size()[1:])
            
            # Convert tensors to NumPy arrays for visualization
            predicted_mask_np = predicted_mask.squeeze().cpu().numpy()
            ground_truth_np = ground_truth_tensor.squeeze().cpu().numpy()
            input_image_np = np.array(image)
            
            # Plot original image, ground truth mask, and predicted mask
            axes[i, 0].imshow(input_image_np)
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(ground_truth_np, cmap="gray")
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis("off")
            
            axes[i, 2].imshow(predicted_mask_np, cmap="gray")
            axes[i, 2].set_title("Predicted Mask")
            axes[i, 2].axis("off")
    
    plt.tight_layout()
    plt.show()

# Call the function with your model and paths
test_two_images(model, test_folder, mask_folder, device)
