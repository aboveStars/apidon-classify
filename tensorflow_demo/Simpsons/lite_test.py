# Import necessary libraries
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import os

# Parameters for processing the images
img_height = 180
img_width = 180

# List of character names for prediction output
class_names = ['abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 
               'barney_gumble', 'bart_simpson', 'carl_carlson', 'charles_montgomery_burns', 
               'chief_wiggum', 'cletus_spuckler', 'comic_book_guy', 'edna_krabappel', 
               'groundskeeper_willie', 'homer_simpson', 'kent_brockman', 'krusty_the_clown', 
               'lenny_leonard', 'lisa_simpson', 'maggie_simpson', 'marge_simpson', 'martin_prince', 
               'mayor_quimby', 'milhouse_van_houten', 'moe_szyslak', 'ned_flanders', 'nelson_muntz', 
               'patty_bouvier', 'principal_skinner', 'professor_john_frink', 'rainier_wolfcastle', 
               'ralph_wiggum', 'selma_bouvier', 'sideshow_bob', 'sideshow_mel', 'snake_jailbird', 
               'waylon_smithers']

# Function to preprocess image: fetch, resize, convert to numpy array
def preprocess_image_tflite(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        # Convert image to RGB, dropping the alpha channel if present
        img = img.convert("RGB")
        img = img.resize((img_height, img_width))
        img_array = np.array(img).astype(np.float32)
        # Normalize if your model was trained with normalization
        # img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Create a batch
        return img_array
    else:
        print("Failed to retrieve the image from the URL.")
        return None
# Preprocess the image
simpsons_url = "https://www.example.com/simpsons_image.png"
img_array = preprocess_image_tflite(simpsons_url)

if img_array is not None:
    # Initialize the TensorFlow Lite interpreter
    TF_MODEL_FILE_PATH = os.path.join(os.getcwd(), "model.tflite")
    interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
    interpreter.allocate_tensors()

    # Retrieve and print model's expected input details
    input_details = interpreter.get_input_details()
    print("Model expected input shape:", input_details[0]['shape'])
    print("Model's expected input type:", input_details[0]['dtype'])

    # Confirm and print the prepared image array's shape and type
    print("Prepared img_array shape:", img_array.shape)
    print("Prepared img_array type:", img_array.dtype)

    # Ensure img_array aligns with the expected input shape and type
    # If necessary, here's where you would adjust the img_array 
    # e.g., img_array = img_array.astype(np.float32) if dtype does not match

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # Retrieve and process the output details
    output_details = interpreter.get_output_details()
    predictions_lite = interpreter.get_tensor(output_details[0]['index'])
    score_lite = tf.nn.softmax(predictions_lite[0])
    top_k_values, top_k_indices = tf.nn.top_k(score_lite, k=5)

    print("Top 5 predictions for the uploaded image:")
    for i in range(5):
        predicted_class_name = class_names[top_k_indices.numpy()[i]]
        probability_percent = top_k_values.numpy()[i] * 100
        print(f"{i+1}: {predicted_class_name}, Probability: {probability_percent:.2f}%")
else:
    print("Image preprocessing was not successful.")