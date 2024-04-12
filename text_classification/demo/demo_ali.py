# Import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data with labels
text_data = ["This is a spam email", "I like this movie!", "Buy now! This deal is awesome"]
labels = ["spam", "positive", "spam"]

# Text pre-processing (can be extended for cleaning and normalization)
def preprocess(text):
  # Convert to lowercase
  text = text.lower()
  # Remove punctuation
  # ...
  return text

# Preprocess data
text_data = [preprocess(text) for text in text_data]

# Feature extraction - count word occurrences
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(text_data)

# Train a model - Multinomial Naive Bayes for simplicity
model = MultinomialNB()
model.fit(features, labels)

# New text for prediction
new_text = "This movie is bad."

# Preprocess new text
new_text_features = vectorizer.transform([preprocess(new_text)])

# Predict the class
predicted_label = model.predict(new_text_features)[0]

# Print the prediction
print(f"Predicted label for '{new_text}': {predicted_label}")
