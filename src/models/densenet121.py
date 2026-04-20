
import torch
import torch.nn as nn
from torchvision import models

class DenseNet121Binary(nn.Module):
    def __init__(self, weights='DEFAULT'):
        super().__init__()
        
        # Use 'weights' instead of deprecated 'pretrained'
        self.model = models.densenet121(weights=weights)
        
        # Replace classifier for binary classification
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, 1)  # Binary output
        
    def forward(self, x):
        return self.model(x)
