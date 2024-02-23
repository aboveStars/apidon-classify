# Import necessary libraries
import tensorflow as tf
from keras.models import load_model
from PIL import Image
from io import BytesIO
import requests
import os

# Parameters for processing the images
batch_size = 32
img_height = 90
img_width = 90

# List of character names for prediction output
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# The modified preprocess_image function
def preprocess_image(image_url):
    # Fetch the image from the URL
    response = requests.get(image_url)
    
    # Ensure the request was successful
    if response.status_code == 200:
        # Open the image directly from the response's bytes
        im = Image.open(BytesIO(response.content))
        # Convert it to RGB to ensure compatibility
        rgb_im = im.convert('RGB')
        
        # Resize the image to match model's expected input dimensions
        img = rgb_im.resize((img_height, img_width))
        # Convert the image to a numpy array and add a batch dimension
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        return img_array
    else:
        print("Failed to retrieve the image from the URL.")
        return None

# Assuming you have the URL and the rest of the code as before
nature_url = "https://images.unsplash.com/photo-1584696049838-8e692282a2e6?q=80&w=1000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTF8fHN0cmVldHxlbnwwfHwwfHx8MA%3D%3D"
img_array = preprocess_image(nature_url)

# Load the pre-trained model
if img_array is not None:
    model_path = "/Users/ali/Documents/Apidon/apidon-classify/keras_demo/nature/model.h5"
    model = load_model(model_path)
# Perform prediction using the model
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    top_k_values, top_k_indices = tf.nn.top_k(score, k=3)

# Display the top 3 predictions with their probabilities
    print("Top 3 Predictions for the uploaded image:")
    for i in range(3):
        # Using .numpy() to extract the values from the tensors
        predicted_class_name = class_names[top_k_indices.numpy()[i]]
        probability_percent = top_k_values.numpy()[i] * 100
        print(f"{i+1}: {predicted_class_name}, Probability: {probability_percent:.2f}%")
else:
    print("Image preprocessing was not successful.")