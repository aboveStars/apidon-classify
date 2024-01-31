import tensorflow as tf
import keras
from keras.models import Sequential
from keras import layers
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from io import BytesIO 
import requests
import numpy as np
from PIL import Image 

# preparing image
url = input("Enter URL: ")

def prepare_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((128, 128))  #128 is the image size that cant be changed due to model training in the same size 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    
    return img_array

img_array = prepare_image_from_url(url)

# Refering output classes manually, for now its safe and simple, later we should review
class_names = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

# h.5 format codes if you want to use h.5 format of the model:
# arch4 = keras.models.load_model(r"C:\Users\mfati\arch4.h5")
# predictions= arch4.predict(img_array)

# tflite model codes if you want to use tflite version of the model::
TF_MODEL_FILE_PATH = r'C:\Users\mfati\arch4.tflite' # you need to change path, r'' may need to change too
interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
classify_lite = interpreter.get_signature_runner('serving_default') #signuture to run the model
predictions = classify_lite(sequential_input=img_array)['dense_1']

#preparing the output for readable possibilities
score_lite = tf.nn.softmax(predictions)
top_k_values, top_k_indices = tf.nn.top_k(predictions, k=5) # k for the number of predictions
probabilities = tf.nn.softmax(predictions[0]) # image should be classify one by one

#output loop
for i in range(5):
    class_name = class_names[top_k_indices[0][i]]
    probability = probabilities[top_k_indices[0][i]] * 100
    print(f"Prediction rank {i+1}: {class_name} with a probability of {probability:.2f}%")

