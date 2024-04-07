import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import json
from torch.nn.functional import softmax

# Load ImageNet class labels
with open('/Users/ali/Documents/Apidon/apidon-classify/pre-trained-pt-models/PreTrainedPytorch/imagenet_class_index.json') as f:
    class_idx = json.load(f)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load an image from a URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

# Replace "image_url" with the actual URL of the image you want to classify
image_url = "https://wallpapers.com/images/featured/flower-pictures-unpxbv1q9kxyqr1d.jpg"
img = load_image_from_url(image_url)  # Load image from URL
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

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

# Predict with each model
for model_name, model_loader in model_loaders.items():
    print(f"\nPredictions from {model_name}:")
    model = model_loader()
    model.eval()

    with torch.no_grad():
        out = model(batch_t)

    probs = softmax(out, dim=1)
    top5_probs, top5_inds = torch.topk(probs, 5)
    for i in range(5):
        print(f"{idx2label[top5_inds[0][i]]}: {top5_probs[0][i].item() * 100:.2f}%")