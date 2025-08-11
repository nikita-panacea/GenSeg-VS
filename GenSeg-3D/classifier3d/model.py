from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple3DClassifier(nn.Module):
    """
    Lightweight 3D classifier that consumes [N, 1, D, H, W] and outputs logits [N, 2].
    """
    def __init__(self, in_channels: int = 1, base: int = 32, num_classes: int = 2):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base, 3, padding=1), nn.BatchNorm3d(base), nn.ReLU(inplace=True),
            nn.Conv3d(base, base, 3, padding=1), nn.BatchNorm3d(base), nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = nn.Sequential(
            nn.Conv3d(base, base * 2, 3, padding=1), nn.BatchNorm3d(base * 2), nn.ReLU(inplace=True),
            nn.Conv3d(base * 2, base * 2, 3, padding=1), nn.BatchNorm3d(base * 2), nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = nn.Sequential(
            nn.Conv3d(base * 2, base * 4, 3, padding=1), nn.BatchNorm3d(base * 4), nn.ReLU(inplace=True),
            nn.Conv3d(base * 4, base * 4, 3, padding=1), nn.BatchNorm3d(base * 4), nn.ReLU(inplace=True),
        )
        self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(base * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.pool2(x)
        x = self.enc3(x)
        x = self.pool3(x).flatten(1)
        logits = self.fc(x)
        return logits



import torch
import torch.nn as nn
from monai.networks.nets.densenet import DenseNet201
 
class DenseNet3D(nn.Module):
    def __init__(self):
        super(DenseNet3D, self).__init__()
        self.base = DenseNet201(
            spatial_dims=3,    
            in_channels=1,      
            out_channels=1,    
            dropout_prob=0
        )
 
    def forward(self, x):
        logits = self.base(x)           # raw scores
        return logits
