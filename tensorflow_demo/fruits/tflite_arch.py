import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import os

# Refering output classes manually, for now its safe and simple, later we should review
class_names = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry'] 

url = input("Url: ")

# Adjusting the input
def prepare_image_from_url(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img = img.convert("RGB")
        img = img.resize((128, 128)) #128 img sizes, cant be changed
        img_array = np.array(img).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)  
        return img_array
    else:
        print("Invalid URL.")
        return None

img_array = prepare_image_from_url(url)

if img_array is not None:
    #for using tflite model
    TF_MODEL_FILE_PATH = os.path.join(os.getcwd(), "arch12.tflite")
    interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
    interpreter.allocate_tensors()
    #input details
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # output details
    output_details = interpreter.get_output_details()
    predictions_lite = interpreter.get_tensor(output_details[0]['index'])
    score_lite = tf.nn.softmax(predictions_lite[0])
    top_k_values, top_k_indices = tf.nn.top_k(score_lite, k=5)

    #printing top 5 probility
    for i in range(5):
        class_name = class_names[top_k_indices.numpy()[i]]
        probability = top_k_values.numpy()[i] * 100
        print(f"{i+1}: {class_name}, Probability: {probability:.2f}%")
else:
    print("Image preprocessing was not successful.")