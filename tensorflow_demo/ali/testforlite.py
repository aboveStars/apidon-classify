import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO
import requests

batch_size = 32
img_height = 180
img_width = 180

class_names = ['abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 'barney_gumble', 'bart_simpson', 'carl_carlson', 'charles_montgomery_burns', 'chief_wiggum', 'cletus_spuckler', 'comic_book_guy', 'edna_krabappel', 'groundskeeper_willie', 'homer_simpson', 'kent_brockman', 'krusty_the_clown', 'lenny_leonard', 'lisa_simpson', 'maggie_simpson', 'marge_simpson', 'martin_prince', 'mayor_quimby', 'milhouse_van_houten', 'moe_szyslak', 'ned_flanders', 'nelson_muntz', 'patty_bouvier', 'principal_skinner', 'professor_john_frink', 'rainier_wolfcastle', 'ralph_wiggum', 'selma_bouvier', 'sideshow_bob', 'sideshow_mel', 'snake_jailbird', 'waylon_smithers']

simpsons_url = "https://www.liveabout.com/thmb/XQoBrEqvUuTivcRlJKkx8ckWIhs=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/homer_2008_v2F_hires2-56a00fd43df78cafda9fde98.jpg"
response = requests.get(simpsons_url)
img = Image.open(BytesIO(response.content))
img = img.resize((img_height, img_width))

# Convert the image to a numpy array
img_array = np.array(img)

# Normalize the image data if needed (ensure it matches the normalization during training)
# Example: img_array = img_array / 255.0

# Ensure the data type is compatible with the TensorFlow Lite model
img_array = img_array.astype(np.float32)

# Expand dimensions to create a batch
img_array = np.expand_dims(img_array, axis=0)

# Continue with the rest of your code

# Run the TensorFlow Lite model
TF_MODEL_FILE_PATH = '/Users/ali/Documents/Apidon/apidon-classify/model.tflite'
interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()

# Get the output
predictions_lite = interpreter.get_tensor(output_details[0]['index'])
score_lite = tf.nn.softmax(predictions_lite)

top_k_values, top_k_indices = tf.nn.top_k(predictions_lite, k=5)
probabilities = tf.nn.softmax(predictions_lite[0])

for i in range(5):
    class_name = class_names[top_k_indices[0][i]]
    probability = probabilities[top_k_indices[0][i]] * 100
    print(f"Prediction rank {i+1}: {class_name} with a probability of {probability:.2f}%")
