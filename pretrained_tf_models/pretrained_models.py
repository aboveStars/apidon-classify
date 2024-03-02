from keras.applications.convnext import ConvNeXtTiny, preprocess_input as pre_conv
from keras.applications.resnet50 import ResNet50, preprocess_input as pre_resnet
from keras.applications.vgg19 import VGG19, preprocess_input as pre_vgg
from keras.applications.mobilenet import  MobileNet, preprocess_input as pre_mobilnet
from keras.applications.densenet import DenseNet201, preprocess_input as pre_densenet
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from PIL import Image
import requests
from io import BytesIO
import numpy as np


image_url = input("enter url:")

#Load pretrained models
resnet_model = ResNet50(weights="imagenet")
vgg_model = VGG19(weights="imagenet")
mobilenet_model = MobileNet(weights="imagenet")
densenet_model = DenseNet201(weights="imagenet")
conv_model_tiny = ConvNeXtTiny(weights="imagenet")


# Prepare image
def preprocess_image(url, model):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))  

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    if model == "resnet":
        x = pre_resnet(x)
    elif model == "vgg":
        x = pre_vgg(x)
    elif model == "mobilenet":
        x = pre_mobilnet(x)
    elif model == "densenet":
        x = pre_densenet(x)
    elif model == "convnext":
        x = pre_conv(x)

    return x


# top 5 prediction
def predict_image(url, model, model_name):
    x = preprocess_image(url, model_name)

    if model_name == "resnet":
        preds = resnet_model.predict(x)
    elif model_name == "vgg":
        preds = vgg_model.predict(x)
    elif model_name == "mobilenet":
        preds = mobilenet_model.predict(x)
    elif model_name == "densenet":
        preds = densenet_model.predict(x)
    if model_name == "convnext":
        preds = conv_model_tiny.predict(x)

    decoded_preds = decode_predictions(preds, top=5)[0]

    print(f"Top 5 Predictions for {model_name}:")
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        print(f"{i+1}: {label} ({score:.2%})")




predict_image(image_url, resnet_model, "resnet")
predict_image(image_url, vgg_model, "vgg")
predict_image(image_url, mobilenet_model, "mobilenet")
predict_image(image_url, densenet_model, "densenet")
predict_image(image_url, conv_model_tiny, "convnext")

