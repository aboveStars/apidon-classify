import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import csv
import urllib.request

task = 'emotion'  # Specify the task (e.g., 'emotion' or any other task available in TweetEval)
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

# Download label mapping
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

def classify_text(text):
    # Tokenize the text
    encoded_input = tokenizer(text, return_tensors='tf')
    # Get model predictions
    output = model(encoded_input)
    scores = output[0][0].numpy()
    scores = softmax(scores)
    # Get the top-ranked labels
    ranking = np.argsort(scores)[::-1]
    top_labels = [(labels[i], round(float(scores[i]), 4)) for i in ranking]
    return top_labels

# Example usage
text_to_classify = "Good night 😊"
predicted_labels = classify_text(text_to_classify)

print(f"Text: {text_to_classify}")
print("Predicted Labels:")
for idx, (label, score) in enumerate(predicted_labels, 1):
    print(f"{idx}) {label}: {score}")
