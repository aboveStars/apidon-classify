from transformers import pipeline

# Load 
reviewer_sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

reviewer_test_cases = [
    "This movie is fantastic!",
    "I wouldn't recommend this product to anyone.",
    "The service at this restaurant was terrible.",
    "I'm very satisfied with my purchase.",
    "The staff was rude and unhelpful."
]


# Perform 
print("Reviewer Sentiment Analysis:")
for i, text in enumerate(reviewer_test_cases, 1):
    result = reviewer_sentiment_analyzer(text)
    print(f"Test Case {i}: {text}")
    print("Sentiment:", result[0]['label'])
    print()
