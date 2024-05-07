from transformers import pipeline

# Load 
classifier = pipeline('sentiment-analysis', framework='tf')

# samples
sentences = [
    "I love this product! It's amazing.",
    "This movie was terrible. I hated it.",
    "The weather today is beautiful.",
    "The customer service was excellent.",
    "I'm feeling really happy today.",
    "The food at the restaurant was delicious.",
    "I had a horrible experience with this company.",
    "The book I read was boring and poorly written.",
    "The new update for the app is fantastic.",
    "I'm disappointed with the service I received."
]

# Classify 
for sentence in sentences:
    result = classifier(sentence)
    label = result[0]['label']
    score = result[0]['score']
    
    print(f"Sentence: {sentence}")
    print(f"Sentiment: {label}, Score: {score:.2f}")
    print()
