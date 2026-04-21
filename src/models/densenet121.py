import torch
import torch.nn as nn
from torchvision import models

class DenseNet121Binary(nn.Module):
    def __init__(self, weights='DEFAULT', dropout_rate=0.5):
        super().__init__()
        
        self.model = models.densenet121(weights=weights)
        
        # Add dropout before classifier
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        return self.model(x)
