import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import nn
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from torchtext.data.functional import to_map_style_dataset

# Define the model architecture
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# Load tokenizer and vocabulary
tokenizer = get_tokenizer("basic_english")
train_iter = AG_NEWS(split="train")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))

# Load saved model
model = TextClassificationModel(len(vocab), 64, 4)  # Assuming 4 classes for AG_NEWS
model.load_state_dict(torch.load("/Users/ali/Documents/Apidon/TextClassification/Pytorch/model.pth"))
model.eval()

# Define label names
ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}

# Prediction function
def predict_class(text):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        predicted_class = output.argmax(1).item()
        return predicted_class + 1  # Adding 1 since AG_NEWS labels start from 1

# Test prediction on sample text
sample_text = "Apple's latest product launch was a huge success."
predicted_label = predict_class(sample_text)
predicted_label_name = ag_news_label[predicted_label]
print(f"Predicted class: {predicted_label_name}")
