import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision

# Load the pre-trained logo model
model_logo = torchvision.models.resnet18()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the state dict and move the model to the chosen device
model_logo.load_state_dict(torch.load(os.getcwd()+'\\trained_model_brands.pth', map_location = device))
model_logo.to(device)

# Load the pre-trained humans model
model_humans = torchvision.models.resnet18()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the state dict and move the model to the chosen device
model_humans.load_state_dict(torch.load(os.getcwd()+'\\trained_model_humans.pth', map_location = device))
model_humans.to(device)

# Set models to evaluation mode
model_logo.eval()
model_humans.eval()

def filter(img_path):
    # Define the transformation to preprocess the image
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Lambda(lambda x: x.convert('RGB') if x.mode == 'RGBA' else x),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Open the image and apply the transformation
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)

    # Move the image to the same device as the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = img.to(device)

    # Run the image through the logo model
    logo_output = model_logo(img)
    _, logo_pred = torch.max(logo_output, 1)
    logo_prob = torch.softmax(logo_output, dim=1)
    logo_prob = logo_prob[0][logo_pred]
    #Run the image through the human model
    human_output = model_humans(img)
    _, human_pred = torch.max(human_output, 1)
    human_prob = torch.softmax(human_output, dim=1)
    human_prob = human_prob[0][human_pred]
    
    # Print the results
    if human_pred == 1 and human_prob > 0.5: # increase in threshold will increase recall/false negatives 
        print("Human detected")
    elif logo_pred == 1 and logo_prob > 0.5: # increase in threshold will increase recall/false negatives 
        print("Logo detected")
    else:
        print("Neither human nor logo detected")
    
# Test the filter function with on 4 images
for i in range(1,5):
    if i<=2:
        filter(os.getcwd() + f"\\test_{i}.jpg") # path to image you want to filter
    else:
        filter(os.getcwd() + f"\\test_{i}.png") # path to image you want to filter
