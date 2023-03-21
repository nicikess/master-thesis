import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet18_Weights


class ResNet:
    def __init__(self, number_of_input_channels, number_of_classes):

        self.number_of_classes = number_of_classes
        self.model = models.resnet18(weights=ResNet18_Weights)
        self.model.conv1 = nn.Conv2d(
            number_of_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        pretrained_bool = True
        if pretrained_bool:
            for param in self.model.parameters():
                param.requires_grad = False

        # Adding two fully connected layers
        self.model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.number_of_classes)
        )
