# Brand Logo and Human Detection Project Pytorch 
## Overview
This project is designed to detect the presence of brand logos and humans in images. Two models were trained using the ResNet18 architecture, one for detecting humans and one for detecting brand logos.The code uses GPU for training and evaluation if available, otherwise it uses CPU.The project includes a total of 5 Python files: train_humans.py, train_logos.py, eval_humans.py, eval_logos.py, and filter.py.

The train_humans.py and train_logos.py files contain the necessary methods and classes for loading and preprocessing the data, as well as training the models. The eval_humans.py and eval_logos.py files then use these methods to train and evaluate the models on a test dataset. Finally, the filter.py file uses the trained models saved on disk to filter images containing either humans or brand logos.

## Usage
To use this project, the user needs to provide the path to the image dataset in the appropriate variables in the train_humans.py and train_logos.py files. The user can then run the two eval files only to train and evaluate the models on a test dataset and get the evaluation metrics.

The user can also use the filter.py file to filter images containing either humans or brand logos by providing the path to the trained models and the directory containing the images to be filtered.

- Make sure to provide the correct paths to the dataset folders containing human images and logo images in the train_humans.py, eval_humans.py, train_logos.py and eval_logos.py files respectively.
- The data should be structured such that the images are organized into sub-folders, each representing a class label.

## Results
The models were trained and evaluated on a dataset of images containing both humans and brand logos. The human detection model achieved an accuracy of 0.8973, precision of 0.9065, recall of 0.8973, and F1 score of 0.8975. The brand logo detection model achieved an accuracy of 0.9076, precision of 0.9086, recall of 0.9076, and F1 score of 0.9077. These results demonstrate that the models are able to effectively detect the presence of humans and brand logos in images.
