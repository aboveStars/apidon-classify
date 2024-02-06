# Import necessary libraries
import tensorflow as tf
from keras.models import load_model
from PIL import Image
from io import BytesIO
import requests
import os

# Parameters for processing the images
batch_size = 32
img_height = 128
img_width = 128

# Refering output classes manually, for now its safe and simple, later we should review
class_names = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']

def preprocess_image(url):
    response = requests.get(url)
    # Ensure the request was successful
    if response.status_code == 200:
        im = Image.open(BytesIO(response.content))
        # Convert it to RGB 
        rgb_im = im.convert('RGB')
        img = rgb_im.resize((128, 128))
       
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        return img_array
    else:
        print("Failed to retrieve the image from the URL.")
        return None

# get url
url = input("enter URL: ")
img_array = preprocess_image(url)

# Load the pre-trained model
if img_array is not None:
    model_path = os.path.join(os.getcwd(), "arch12.h5")
    model = load_model(model_path)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    top_k_values, top_k_indices = tf.nn.top_k(score, k=5)

# Display the top 5 predictions
    print("Top 5 Predictions for the uploaded image:")
    for i in range(5):
        predicted_class_name = class_names[top_k_indices.numpy()[i]]
        probability_percent = top_k_values.numpy()[i] * 100
        print(f"{i+1}: {predicted_class_name}, Probability: {probability_percent:.2f}%")
else:
    print("Image preprocessing was not successful.")