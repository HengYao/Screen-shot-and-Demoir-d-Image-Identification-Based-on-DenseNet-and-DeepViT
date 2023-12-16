import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch.deepvit import DeepViT
from densnet201 import *
import warnings
warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2 = nn.Conv2d(3, 9, kernel_size=5, stride=1, padding=2, bias=False)
        self.pool = nn.AvgPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(9)
        self.relu = nn.PReLU()
        self.densnet = DenseNet201()
        self.v = v = DeepViT(
            image_size=256,
            patch_size=16,
            num_classes=1000,
            dim=1024,
            depth=1,
            heads=32,
            mlp_dim=256,
            dropout=0.9,
            emb_dropout=0.9
        )
        self.flatten = nn.Flatten()
        self.LC1 = nn.Linear(1000, 2048)
        self.LC5 = nn.Linear(2048,1024)
        self.LC4 = nn.Linear(1024, 2)

    def forward(self, x):
        x = x.to(device)
        x = self.densnet(x)
        x = self.bn1(x)
        x = self.v(x)
        x = self.flatten(x)
        x = self.LC1(x)
        x = F.dropout(x,p=0.9)
        x = self.relu(x)
        x = self.LC5(x)
        x = F.dropout(x,p=0.9)
        x = self.relu(x)
        x = self.LC4(x)
        x = torch.softmax(x, 1)
        return x
