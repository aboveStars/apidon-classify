import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch import nn
from torchtext.data.functional import to_map_style_dataset

# Load AG News dataset
train_iter = AG_NEWS(split="train")

# Prepare data processing pipeline
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

# Generate data batch and iterator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets

# Define TextClassificationModel
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

    def forward(self, text):
        embedded = self.embedding(text)
        return self.fc(embedded)

# Load the saved model and make predictions
model = TextClassificationModel(len(vocab), 64, 4)  # Assuming 4 classes for AG News
model.load_state_dict(torch.load("/Users/ali/Documents/Apidon/TextClassification/Pretrainedpyt/model.pth"))
model.eval()

def predict_class(text):
    text = torch.tensor(text_pipeline(text), dtype=torch.int64).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(text)
        predicted_class = output.argmax(1).item() + 1  # Adding 1 to match AG News labels
    return predicted_class

# Example usage
text = "Microsoft unveils new AI-powered tools for developers"
predicted_class = predict_class(text)
print(f"Predicted class: {predicted_class}")
