import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

# Test cases
test_cases = [
    "I love this movie! It's amazing.",
    "This movie is terrible.",
    "The food at this restaurant is delicious.",
    "I cannot believe how bad this product is.",
    "The customer service was excellent."
]

for i, text in enumerate(test_cases, 1):
    
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)

    predicted_label = torch.argmax(outputs.logits, dim=1).item()

    if(predicted_label == 1):
      result = 'postive'
    else:
      result = 'negative'

    # Print 
    print(f"Test Case {i}: {text}")
    print("Predicted Sentiment Label:", result)
