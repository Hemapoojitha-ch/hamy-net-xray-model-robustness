
import torch
import torch.nn as nn
from torchvision import models

class DenseNet121Binary(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        self.model = models.densenet121(pretrained=pretrained)
        
        # Replace classifier
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, 1)  # Binary output
        
    def forward(self, x):
        return self.model(x)
