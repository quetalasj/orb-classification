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

class Regressor(BaseModel):
    def __init__(self, in_size, hidden_sizes, out_size, train_backbone=True):
        super().__init__()
        self.transform = DEFAULT_RESNET_TRANSFORM
        self.activation = nn.Sigmoid()
        self.loss = nn.MSELoss()

        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.backbone.fc = nn.Identity()
        self.fc = get_fc_layers(in_size, hidden_sizes, out_size)
        print("network head")
        print(self.fc)
        self.train_backbone(train_backbone)
        # self.save_hyperparameters()

    def forward(self, x):
        y = self.backbone(x)
        return self.fc(y)
