from torch import nn
from .resnet import ResNet
import numpy as np
import torch


class DualResNet(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, number_of_classes):
        super(DualResNet, self).__init__()

        # First stream of ResNet() for Sentinel 1 data (in_channels_1 = 2)
        self.res_net_1 = ResNet(in_channels_1, number_of_classes).model
        # Second stream of ResNet() for Sentinel 2 data (in_channels_2 = 3)
        self.res_net_2 = ResNet(in_channels_2, number_of_classes).model

        # TODO: Uncomment to test
        #self.classifier = nn.Linear(256, number_of_classes)


    def forward(self, x1, x2):
        # We process Sentinel1 input
        x1 = self.res_net_1(x1)
        # We process Sentinel2 input
        x2 = self.res_net_2(x2)

        tensor_sum = torch.add(x1, x2)
        tensor_mean = tensor_sum / 2

        # TODO: Uncomment to test
        #fusion_output = torch.cat((x1, x2), dim=1)
        # Pass the fused output through the classification layer
        #final_output = self.classifier(fusion_output)

        return tensor_mean
