from train_humans import train_model, evaluate_model
import numpy as np
import torchvision
import torch
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set Arguments for train_model() function
root_dir = os.getcwd()+ '\\humans' # path to humans dataset
batch_size = 64
num_epochs = 10

# Call the train_model function to train the model
device, test_data_loader = train_model(root_dir, batch_size, num_epochs)

# Load the model from disk
model = torchvision.models.resnet18()

# Check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the state dict and move the model to the chosen device
model.load_state_dict(torch.load(os.getcwd()+'\\trained_model_humans.pth', map_location = device))
model.to(device)

# Evaluate the model on the test dataset
evaluate_model(model, test_data_loader, device)