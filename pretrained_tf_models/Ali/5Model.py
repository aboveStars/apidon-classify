# Import necessary libraries
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet
from keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg
from keras.applications.mobilenet import (
    MobileNet,
    preprocess_input as preprocess_mobilenet,
)
from keras.applications.nasnet import (
    NASNetMobile,
    preprocess_input as preprocess_nasnet,
)
from keras.applications.densenet import (
    DenseNet201,
    preprocess_input as preprocess_densenet,
)
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
nasnet_model = NASNetMobile(
    weights="imagenet"
)  # Use NasNetMobile instead of InceptionResNetV2
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
    # Add more conditions if you have other models

    return x


# Function to predict and print top 5 predictions for a given model
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
    # Add more conditions if you have other models

    # Decode predictions and print top 5
    decoded_preds = decode_predictions(preds, top=5)[0]
    print(f"Top 5 Predictions for {model_name}:")
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        print(f"{i+1}: {label} ({score:.2%})")


# Example usage
image_url = "https://images.pexels.com/photos/1207875/pexels-photo-1207875.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500"
predict_image(image_url, resnet_model, "resnet")
predict_image(image_url, vgg_model, "vgg")
predict_image(image_url, mobilenet_model, "mobilenet")
predict_image(image_url, nasnet_model, "nasnet")
predict_image(image_url, densenet_model, "densenet")

# Example with path
# file_path="/"
# predict_image(file_path, resnet_model, "resnet")
# predict_image(file_path, vgg_model, "vgg")
# predict_image(file_path, mobilenet_model, "mobilenet")
# predict_image(file_path, nasnet_model, "nasnet")
# predict_image(file_path, densenet_model, "densenet")

# Add more calls if you have other models
