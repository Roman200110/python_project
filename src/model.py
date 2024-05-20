import torch
import torch.nn as nn
import torch.hub


class AnimalClassifier(nn.Module):
    """
    A class used to represent the CNN model for animal classification.

    Methods
    -------
    forward(x):
        Defines the forward pass of the model.
    """

    def __init__(self, num_classes: int = 4):
        super(AnimalClassifier, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
