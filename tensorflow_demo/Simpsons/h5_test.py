# Import necessary libraries
import tensorflow as tf
from keras.models import load_model
from PIL import Image
from io import BytesIO
import requests
import os

# Parameters for processing the images
batch_size = 32
img_height = 180
img_width = 180

# List of character names for prediction output
class_names = ['abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 'barney_gumble', 'bart_simpson', 'carl_carlson', 'charles_montgomery_burns', 'chief_wiggum', 'cletus_spuckler', 'comic_book_guy', 'edna_krabappel', 'groundskeeper_willie', 'homer_simpson', 'kent_brockman', 'krusty_the_clown', 'lenny_leonard', 'lisa_simpson', 'maggie_simpson', 'marge_simpson', 'martin_prince', 'mayor_quimby', 'milhouse_van_houten', 'moe_szyslak', 'ned_flanders', 'nelson_muntz', 'patty_bouvier', 'principal_skinner', 'professor_john_frink', 'rainier_wolfcastle', 'ralph_wiggum', 'selma_bouvier', 'sideshow_bob', 'sideshow_mel', 'snake_jailbird', 'waylon_smithers']

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
simpsons_url = "https://www.example.com/simpsons_image.png"
img_array = preprocess_image(simpsons_url)

# Load the pre-trained model
if img_array is not None:
    model_path = os.path.join(os.getcwd(), "model.h5")
    model = load_model(model_path)
# Perform prediction using the model
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    top_k_values, top_k_indices = tf.nn.top_k(score, k=5)

# Display the top 5 predictions with their probabilities
    print("Top 5 Predictions for the uploaded image:")
    for i in range(5):
        # Using .numpy() to extract the values from the tensors
        predicted_class_name = class_names[top_k_indices.numpy()[i]]
        probability_percent = top_k_values.numpy()[i] * 100
        print(f"{i+1}: {predicted_class_name}, Probability: {probability_percent:.2f}%")
else:
    print("Image preprocessing was not successful.")