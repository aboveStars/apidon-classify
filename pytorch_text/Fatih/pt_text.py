import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle

# Sizes
input_size=467
output_size=2
hidden_size=500

# model arch
class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.fc1 = torch.nn.Linear(input_size, hidden_size)
       self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
       self.fc3 = torch.nn.Linear(hidden_size, output_size)

   def forward(self, X):
       X = torch.relu((self.fc1(X)))
       X = torch.relu((self.fc2(X)))
       X = self.fc3(X)
       return F.log_softmax(X,dim=1)


# Load the model
model = Net()
model.load_state_dict(torch.load('text_classifier_pytorch'))

# Load the text vectorizer
with open('pt11.pickle', 'rb') as file:
    vectorizer = pickle.load(file)

# Preprocess the text
def predict_sentiment(samples):
    samples_preprocessed = vectorizer.transform(samples).toarray()
    samples_tensor = torch.from_numpy(samples_preprocessed).float()

    # Make predictions
    with torch.no_grad():
        output = model(samples_tensor)
        _, predicted = torch.max(output, 1)
    
    # Convert to class
    class_labels = ['Negative', 'Positive']
    predictions = [class_labels[pred] for pred in predicted]

    return predictions

# Example cases
new_samples = [
    "The staff was friendly and attentive.",
    "The restaurant was dirty and poorly managed.",
    "The service was slow and unfriendly.",
    "The meal was excellent, but the prices were too high.",
    "The ambiance was pleasant, but the food was disappointing.",
    "Overall, a great dining experience.",
    "Would not recommend this restaurant to anyone."
]



predictions = predict_sentiment(new_samples)
for sample, prediction in zip(new_samples, predictions):
    print(f"Sample: '{sample}' => Prediction: {prediction}")

