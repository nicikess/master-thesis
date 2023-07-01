from torch import nn
import numpy as np
import torch
import torchvision.models as models

class DualResNet(nn.Module):
    def __init__(self, state_dict, in_channels_1, in_channels_2, number_of_classes):
        super(DualResNet, self).__init__()

        # First stream of ResNet()
        self.res_net_1 = ResNet(state_dict["state_dict_modality_1"], in_channels_1, number_of_classes).model
        # Second stream of ResNet()
        self.res_net_2 = ResNet(state_dict["state_dict_modality_2"], in_channels_2, number_of_classes).model

        # TODO: Uncomment to test
        self.fc = LinearFC(2 * 256, number_of_classes)

    def forward(self, x1, x2):
        # Process modality 1 input
        x1 = self.res_net_1(x1)
        # Process modality 2 input
        x2 = self.res_net_2(x2)

        x = torch.cat([x1, x2], dim=1)

        x = self.fc(x)

        return x

class ResNet(nn.Module):

    def __init__(self, state_dict_from_checkpoint, in_channels_1, number_of_classes):
        super(ResNet, self).__init__()
        self.number_of_classes = number_of_classes
        self.model = models.resnet18(weights=None)

        self.model.conv1 = nn.Conv2d(
            in_channels_1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Update weights
        common_keys = set(self.model.state_dict().keys()) & set(state_dict_from_checkpoint.keys())
        new_state_dict = {k: v for k, v in state_dict_from_checkpoint.items() if k in common_keys}
        self.model.load_state_dict(new_state_dict, strict=False)

        # Check if weights are initialized correctly
        state_dict1 = self.model.state_dict()
        state_dict2 = state_dict_from_checkpoint

        for key1, value1 in state_dict1.items():
            if key1 in state_dict2:
                value2 = state_dict2[key1]
                if torch.allclose(value1, value2):
                    print(f"Weights for key '{key1}' are the same.")
                else:
                    print(f"Weights for key '{key1}' are different.")
            else:
                print(f"Key '{key1}' does not exist in the second network's state dictionary.")

        for key2 in state_dict2.keys():
            if key2 not in state_dict1:
                print(f"Key '{key2}' does not exist in the first network's state dictionary.")

        # Adding two fully connected layers
        self.model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
    
    def forward(self, x):
        return self.model.forward(x)


# Create the last convolution block
class LinearFC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearFC, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fc(x)