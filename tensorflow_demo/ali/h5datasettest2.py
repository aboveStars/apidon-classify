import tensorflow as tf 
from keras.models import load_model
from PIL import Image
import numpy as np
from io import BytesIO
import requests

batch_size = 32
img_height = 180
img_width = 180

class_names = ['abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 'barney_gumble', 'bart_simpson', 'carl_carlson', 'charles_montgomery_burns', 'chief_wiggum', 'cletus_spuckler', 'comic_book_guy', 'edna_krabappel', 'groundskeeper_willie', 'homer_simpson', 'kent_brockman', 'krusty_the_clown', 'lenny_leonard', 'lisa_simpson', 'maggie_simpson', 'marge_simpson', 'martin_prince', 'mayor_quimby', 'milhouse_van_houten', 'moe_szyslak', 'ned_flanders', 'nelson_muntz', 'patty_bouvier', 'principal_skinner', 'professor_john_frink', 'rainier_wolfcastle', 'ralph_wiggum', 'selma_bouvier', 'sideshow_bob', 'sideshow_mel', 'snake_jailbird', 'waylon_smithers']

simpsons_url = "https://cdn.britannica.com/47/101047-050-10641549/Simpsons---Lisa-Maggie-Bart-Marge-Homer.jpg"
response = requests.get(simpsons_url)
img = Image.open(BytesIO(response.content))
img = img.resize((img_height, img_width))

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

model_path = '/Users/ali/Documents/Apidon/apidon-classify/model.h5'
model = load_model(model_path)

predictions = model.predict(img_array)
score_lite = tf.nn.softmax(predictions)

probabilities = tf.nn.softmax(predictions[0])
top_k_values, top_k_indices = tf.nn.top_k(predictions, k=5)

print("Top 5 Predictions for the uploaded image:")
for i in range(5):
    # Fetch the class name using indices obtained from top_k_indices
    predicted_class_name = class_names[top_k_indices[0][i]]
    # Convert the probability to a percentage
    probability_percent = top_k_indices[0][i] * 100
    
    print(f"{i+1}: {predicted_class_name}, Probability: {probability_percent:.2f}%")
