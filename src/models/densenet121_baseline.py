import torch
import torch.nn as nn
from torchvision import models

class DenseNet121Baseline(nn.Module):
    """Vanilla DenseNet121 - minimal changes"""
    def __init__(self, weights='DEFAULT'):
        super().__init__()
        
        self.model = models.densenet121(weights=weights)
        
        # Only change classifier for binary output
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, 1)
        
    def forward(self, x):
        return self.model(x)
