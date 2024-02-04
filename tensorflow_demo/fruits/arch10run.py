import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import os

# Refering output classes manually, for now its safe and simple, later we should review
class_names = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry', 'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Corn Husk', 'Cucumber Ripe', 'Cucumber Ripe 2', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 'Peach Flat', 'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pepino', 'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon', 'Tomato Yellow', 'Tomato not Ripened', 'Walnut', 'Watermelon']
#getting url
url = input("Enter URL: ")

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
    TF_MODEL_FILE_PATH = os.path.join(os.getcwd(), "arch10.tflite")
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


# Here h.5 codes to run if needed
#     if img_array is not None:
#     model_path = os.path.join(os.getcwd(), "arch10.h5")
#     model = load_model(model_path)

#     predictions = model.predict(img_array)
#     score = tf.nn.softmax(predictions[0])
#     top_k_values, top_k_indices = tf.nn.top_k(score, k=5)

    #printing top 5 probility
    for i in range(5):
        class_name = class_names[top_k_indices.numpy()[i]]
        probability = top_k_values.numpy()[i] * 100
        print(f"{i+1}: {class_name}, Probability: {probability:.2f}%")
else:
    print("Image preprocessing was not successful.")