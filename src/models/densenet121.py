import torch
import torch.nn as nn
from torchvision import models

class DenseNet121Binary(nn.Module):
    def __init__(self, weights='DEFAULT', freeze_backbone=False):
        super().__init__()
        
        self.model = models.densenet121(weights=weights)
        
        # Optionally freeze backbone for better training stability
        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False
        
        # Add dropout before classifier
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1)
        )
        
    def forward(self, x):
        return self.model(x)
