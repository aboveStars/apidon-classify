import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import json
from torch.nn.functional import softmax

# Assuming this function fetches your preprocessed tensor
def get_preprocessed_image_tensor(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t

image_url = "https://wallpapers.com/images/featured/flower-pictures-unpxbv1q9kxyqr1d.jpg"
image_tensor = get_preprocessed_image_tensor(image_url)

# List of models you want to use
# Define model loading functions
def load_efficientnet_v2_s():
    return models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

def load_convnext_tiny():
    return models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

def load_swin_transformer_tiny():
    return models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)

def load_mobilenet_v2():
    # Using the new weights parameter instead of pretrained=True
    return models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

def load_resnet50():
    return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Model loaders map
model_loaders = {
    'EfficientNet_V2_S': load_efficientnet_v2_s,
    'ConvNeXt_Tiny': load_convnext_tiny,
    'Swin_Transformer_Tiny': load_swin_transformer_tiny,
    'MobileNetV2': load_mobilenet_v2,
    'ResNet50': load_resnet50,
}
model_list = [models.efficientnet_v2_s, models.convnext_tiny, models.swin_t, models.mobilenet_v2, models.resnet50]

# Load ImageNet class labels
with open('imagenet_class_index.json') as f:
    idx2label = [labels[1] for labels in json.load(f).values()]

# Aggregate predictions
consolidated_predictions = {}

for model_func in model_list:
    model = model_func(weights='DEFAULT').eval()
    with torch.no_grad():
        out = model(image_tensor)
    probs = softmax(out, dim=1)
    top_probs, top_classes = torch.topk(probs, 5)

    for i in range(top_probs.size(1)):
        class_name = idx2label[top_classes[0][i]]
        prob = top_probs[0][i].item() * 100

        # Update if this class's prob is higher than what we've previously encountered
        if class_name not in consolidated_predictions or prob > consolidated_predictions[class_name][1]:
            consolidated_predictions[class_name] = (model_func.__name__, prob)

# Sort and display
sorted_predictions = sorted(consolidated_predictions.items(), key=lambda item: item[1][1], reverse=True)

for class_name, (model_name, prob) in sorted_predictions:
    print(f"{class_name}: {prob:.2f}%")