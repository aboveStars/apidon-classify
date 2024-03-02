# Import necessary libraries
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet
from keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg
from keras.applications.mobilenet import MobileNet, preprocess_input as preprocess_mobilenet
from keras.applications.nasnet import NASNetMobile, preprocess_input as preprocess_nasnet
from keras.applications.densenet import DenseNet201, preprocess_input as preprocess_densenet
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from PIL import Image
import requests
from io import BytesIO
import numpy as np

# Load pretrained models
resnet_model = ResNet50(weights="imagenet")
vgg_model = VGG16(weights="imagenet")
mobilenet_model = MobileNet(weights="imagenet")
nasnet_model = NASNetMobile(weights="imagenet")  
densenet_model = DenseNet201(weights="imagenet")

# Function to preprocess an image for a specific model
def preprocess_image(url, model):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # Resize image based on the model requirements
    if model == "nasnet":
        img = img.resize((224, 224))  # Adjust the size for NasNetMobile
    else:
        img = img.resize((224, 224))  # Default size for other models

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Preprocess input based on the model
    if model == "resnet":
        x = preprocess_resnet(x)
    elif model == "vgg":
        x = preprocess_vgg(x)
    elif model == "mobilenet":
        x = preprocess_mobilenet(x)
    elif model == "nasnet":
        x = preprocess_nasnet(x)
    elif model == "densenet":
        x = preprocess_densenet(x)

    return x

# Function to predict and return top predictions for a given model
def predict_image(url, model, model_name):
    x = preprocess_image(url, model_name)

    # Predict using the appropriate model
    if model_name == "resnet":
        preds = resnet_model.predict(x)
    elif model_name == "vgg":
        preds = vgg_model.predict(x)
    elif model_name == "mobilenet":
        preds = mobilenet_model.predict(x)
    elif model_name == "nasnet":
        preds = nasnet_model.predict(x)
    elif model_name == "densenet":
        preds = densenet_model.predict(x)

    # Decode predictions and return top 5
    decoded_preds = decode_predictions(preds, top=5)[0]
    return decoded_preds

# Function to combine and print predictions for all models
def combine_and_print_predictions(image_url):
    resnet_preds = predict_image(image_url, resnet_model, "resnet")
    vgg_preds = predict_image(image_url, vgg_model, "vgg")
    mobilenet_preds = predict_image(image_url, mobilenet_model, "mobilenet")
    nasnet_preds = predict_image(image_url, nasnet_model, "nasnet")
    densenet_preds = predict_image(image_url, densenet_model, "densenet")

    combined_predictions = {}

    # Combine predictions
    all_preds = [mobilenet_preds, nasnet_preds, densenet_preds, vgg_preds, resnet_preds]
    for preds, model_name in zip(all_preds, ["mobilenet", "nasnet", "densenet", "vgg", "resnet"]):
        for i, (_, label, score) in enumerate(preds):
            if label not in combined_predictions or score > combined_predictions[label]["score"]:
                combined_predictions[label] = {"score": score}

    # Print combined predictions
    sorted_predictions = sorted(combined_predictions.items(), key=lambda x: x[1]["score"], reverse=True)
    print("Combined Predictions:")
    for i, (label, info) in enumerate(sorted_predictions):
        print(f"{i+1}: {label} - {info['score']:.2%}")

# Example usage
image_url = "https://media.istockphoto.com/id/155439315/photo/passenger-airplane-flying-above-clouds-during-sunset.jpg?s=612x612&w=0&k=20&c=LJWadbs3B-jSGJBVy9s0f8gZMHi2NvWFXa3VJ2lFcL0="
combine_and_print_predictions(image_url)
