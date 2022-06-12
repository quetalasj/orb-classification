from .base_model import BaseModel
import torch
import torch.nn as nn
from torchvision import transforms
from .utils import get_fc_layers


DEFAULT_RESNET_TRANSFORM = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class Resnet18Model(BaseModel):
    def __init__(self, in_size, hidden_sizes, out_size, transform=DEFAULT_RESNET_TRANSFORM, freeze_backbone=False, nn_type='classificator'):
        super().__init__(transform, nn_type=nn_type)
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.backbone.fc = nn.Identity()
        self.nn_type = nn_type
        self.fc = get_fc_layers(in_size, hidden_sizes, out_size)
        print("network head")
        print(self.fc)
        self.prepare_backbone(freeze_backbone)
        # self.save_hyperparameters()

    def forward(self, x):
        y = self.backbone(x)
        return self.fc(y)
