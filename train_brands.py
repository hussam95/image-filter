import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import numpy as np

class LogoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset
        
        Parameters:
            root_dir (str): Path to the folder containing the images
            transform (callable, optional): Optional transform to be applied on the images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.files = []
        for c in self.classes:
            c_dir = os.path.join(root_dir, c)
            for file in os.listdir(c_dir):
                self.files.append((os.path.join(c_dir, file), self.class_to_idx[c]))

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Get the image and label at the given index
        
        Parameters:
            idx (int): Index of the image
        
        Returns:
            tuple: Tuple containing the image and the label
        """
        img_path, label = self.files[idx]
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        return image, label

def train_model(root_dir, batch_size, num_epochs):
    """
    Train a model to classify images as containing humans or not
    
    Parameters:
        root_dir (str): Path to the folder containing the images
        batch_size (int): Batch size for training
        num_epochs (int): Number of training epochs
    """

    # Define the transformation to preprocess the images
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Lambda(lambda x: x.convert('RGB') if x.mode == 'RGBA' else x),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Create instances of the custom dataset and the data loader
    dataset = LogoDataset(root_dir, transform)
    
    # Define the split ratio for training and testing
    train_ratio = 0.8
    test_ratio = 0.2

    # Split the dataset into training and testing sets
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    print(f"Train Size: {train_size}; Test Size: {test_size}")

    # Define the data loaders for training and testing
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    #data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define the model architecture and refer to the most up-to-date weights.
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

    # Move the model to the GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in use: {device}")
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Set the model to training mode
    model.train()

    # Train the model for a certain number of epochs
    for epoch in range(num_epochs):
        for images, labels in train_data_loader:
            
            # Move the data to the chosen device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            output = model(images)

            # Compute the loss
            loss = criterion(output, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the average loss for the epoch
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # Save the trained model to disk
    torch.save(model.state_dict(), 'trained_model_brands.pth')
    return device, test_data_loader


# Define the function to evaluate the model
def evaluate_model(model, data_loader, device):
    """
    Evaluate the model's performance on a given dataset
    
    Parameters:
        model (nn.Module): The trained model
        data_loader (DataLoader): The data loader for the dataset to evaluate on
        device (torch.device): The device to run the model on (GPU or CPU)
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize lists to store the true labels and predicted labels
    true_labels = []
    pred_labels = []
    
    # Iterate over the data in the data loader
    with torch.no_grad():
        for images, labels in data_loader:
            # Move the data to the chosen device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            output = model(images)
            
            # Get the predicted labels
            _, pred = torch.max(output, 1)
            
            # Convert the labels and predictions to numpy arrays
            labels = labels.cpu().numpy()
            pred = pred.cpu().numpy()
            
            # Append the labels and predictions to the lists
            true_labels.append(labels)
            pred_labels.append(pred)
    
    # Concatenate the lists of labels and predictions
    true_labels = np.concatenate(true_labels)
    pred_labels = np.concatenate(pred_labels)
    
    # Calculate the performance metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    # Print the performance metrics
    print("\n")
    print(f'Accuracy: {accuracy*100:.2f} %')
    print(f'Precision: {precision*100:.2f} %')
    print(f'Recall: {recall*100:.2f} %')
    print(f'F1 Score: {f1*100:.2f} %')
