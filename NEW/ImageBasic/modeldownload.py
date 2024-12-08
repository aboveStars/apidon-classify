import torch
import torchvision.models as models

model = models.inception_v3(pretrained=True)

save_path = ''

torch.save(model.state_dict(), save_path)