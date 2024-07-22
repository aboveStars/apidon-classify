import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests
from io import BytesIO
import json
import numpy as np

# Function to load labels from a JSON file
def load_labels(label_file):
    with open(label_file, "r") as f:
        labels_data = json.load(f)
    labels = labels_data.get('labels', None)
    if labels is None:
        raise ValueError("Labels not found in the JSON file")
    return labels

# General function to preprocess an image
def preprocess_image(image_url, img_height, img_width, preprocess_function=None):
    response = requests.get(image_url)
    
    if response.status_code == 200:
        im = Image.open(BytesIO(response.content)).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor()
        ])
        
        img_tensor = transform(im).unsqueeze(0)
        
        if preprocess_function:
            img_tensor = preprocess_function(img_tensor)
        return img_tensor
    else:
        print("Failed to retrieve the image from the URL.")
        return None

# Function to get the preprocessing function
def get_preprocessing_function():
    preprocess_function = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
    return preprocess_function

# Function to get the input size of the model
def get_model_input_size():
    img_height, img_width = 299, 299  # InceptionV3 default input size
    return img_height, img_width

# Function to predict using a model
def predict_image(model_path, label_file, image_url):
    # Load the model
    model = models.inception_v3(weights=None, init_weights=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    preprocess_function = get_preprocessing_function()
    img_height, img_width = get_model_input_size()
    
    img_tensor = preprocess_image(image_url, img_height, img_width, preprocess_function)
    
    if img_tensor is not None:
        labels = load_labels(label_file)
        with torch.no_grad():
            predictions = model(img_tensor)
            probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
            top_k_values, top_k_indices = torch.topk(probabilities, k=10)
        
        print("Top 10 Predictions for the uploaded image:")
        for i in range(10):
            predicted_class_name = labels[top_k_indices[i].item()]
            probability_percent = top_k_values[i].item() * 100
            print(f"{i+1}: {predicted_class_name}, Probability: {probability_percent:.2f}%")
    else:
        print("Image preprocessing was not successful.")

if __name__ == "__main__":
    model_path = input("Enter the model path: ").strip()
    label_file = input("Enter the label file path: ").strip()
    image_url = input("Enter the image URL: ").strip()
    
    predict_image(model_path, label_file, image_url)
