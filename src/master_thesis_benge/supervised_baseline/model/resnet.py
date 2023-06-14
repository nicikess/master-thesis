import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, weights, in_channels_1, number_of_classes):
        super(ResNet, self).__init__()
        self.number_of_classes = number_of_classes
        self.model = models.resnet18(weights=weights)
        # wandb.log({"Model size": str(self.model)})
        self.model.conv1 = nn.Conv2d(
            in_channels_1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Adding two fully connected layers
        self.model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.number_of_classes),
        )
    def forward(self, x):
        return self.model.forward(x)


class UniResNet(nn.Module):
    def __init__(self, in_channels_1, number_of_classes):
        super(UniResNet, self).__init__()

        # First stream of ResNet()
        self.res_net_1 = ResNet(in_channels_1, number_of_classes).model

    def forward(self, x1):
        # Process modality 1 input
        x = self.res_net_1(x1)

        return x