import torchvision.models as models
from torchvision import transforms
import torch
from PIL import Image
import requests
from io import BytesIO

# URL
image_url = input("Enter URL: ") 

# Classes for outputs
class_labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = requests.get(class_labels_url)
class_labels = response.text.split('\n')

# Load pre-trained models
squeezenet = models.squeezenet1_0(pretrained=True)
densenet = models.densenet161(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)

image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def classify(model, image_transforms, image_url, topk=5):
    model = model.eval()  
    try:
        # Download and process the image
        response = requests.get(image_url)
        response.raise_for_status()  
        image = Image.open(BytesIO(response.content))

        # Apply transformations 
        image = image_transforms(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():  
            output = model(image)

        # Get the top k predictions and their indices
        probs, indices = torch.topk(output, topk)

        # Print the top k predictions
        print(f"Top {topk} predictions for {model.__class__.__name__}:")
        for i in range(topk):
            class_idx = indices[0][i].item()
            class_prob = probs[0][i].item()
            class_name = class_labels[class_idx]
            print(f"{i+1}: {class_name} with probability {class_prob:.2f}")

    except requests.exceptions.RequestException as e:
        print("Error downloading the image:", e)
    except Exception as e:
        print("Error processing the image:", e)


# Calling models
classify(squeezenet, image_transforms, image_url, topk=5)
classify(densenet, image_transforms, image_url, topk=5)
classify(googlenet, image_transforms, image_url, topk=5)
classify(shufflenet, image_transforms, image_url, topk=5)
classify(resnext50_32x4d, image_transforms, image_url, topk=5)
