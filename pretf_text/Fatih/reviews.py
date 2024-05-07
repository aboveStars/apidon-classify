from transformers import pipeline

# Load
classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment", framework='tf')

# Samples
reviews = [
    "This app is amazing! It has all the features I need and runs smoothly.",
    "Terrible app! It crashes every time I try to use it.",
    "Decent app, but could use some improvements.",
    "The app is okay, but it's a bit slow and laggy. "
]

# Classify
for review in reviews:
    result = classifier(review)
    label = result[0]['label']
    score = result[0]['score']
    print(f"Review: {review}")
    print(f"Star Rating: {label}, Score: {score:.2f}")
    print()
