import torch
import torch.nn as nn
from torchvision import models

class DenseNet121(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(DenseNet121, self).__init__()

        self.model = models.densenet121(pretrained=pretrained)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
