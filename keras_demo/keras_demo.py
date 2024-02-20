import numpy as np
import requests
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
from io import BytesIO

# Define your classes
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

def predict_from_url(model, url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
            
            img = img.resize((64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.  

            # Predict
            preds = model.predict(img_array)
            probs = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)
            
            top_5_indices = np.argsort(probs[0])[::-1][:5]
            for i in top_5_indices:
                print(f"Class {classes[i]}: Probability {probs[0][i]:.2f}")
        else:
            print(f"Error fetching image from URL. Status Code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Load model
try:
    model = load_model('model_keras2.h5')
except Exception as e:
    print(f"Error loading model: {e}")
else:
    url = input("Enter url: ")
    predict_from_url(model, url)