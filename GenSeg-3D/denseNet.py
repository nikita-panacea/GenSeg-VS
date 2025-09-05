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