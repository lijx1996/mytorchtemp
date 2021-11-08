import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import sys
sys.path.append('../')
sys.path.append('./')
from base import BaseModel


class Res50(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.net = torchvision.models.resnet50(pretrained=True)
        self.net.fc = nn.Linear(2048, num_classes, bias=True)
    def forward(self, x):
        x = self.net(x)
        return x

class Res34(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.module = torchvision.models.resnet34(pretrained=False)
        self.module.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.module.fc = nn.Linear(512, num_classes, bias=True)
    def forward(self, x):
        x = self.module(x)
        return x

class Res18(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.module = torchvision.models.resnet18(pretrained=False)
        self.module.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.module.fc = nn.Linear(self.module.fc.in_features, num_classes, bias=True)

    def forward(self, x):
        x = self.module(x)
        return x