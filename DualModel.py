from torch import nn
from Model import ResNetModel
import torch


class DualResNet(nn.Module):

    def __init__(self, in_channels_1, in_channels_2, number_of_classes):
        super(DualResNet, self).__init__()

        # First stream of ResNet() for Sentinel 1 data (in_channels_1 = 2)
        self.res_net_1 = ResNetModel(in_channels_1, number_of_classes)
        # Second stream of ResNet() for Sentinel 2 data (in_channels_2 = 4)
        self.res_net_2 = ResNetModel(in_channels_2, number_of_classes)

        # !!! MAYBE NOT NECESSARY AND NOT CORRECT -> TEST OUT !!!
        self.out_conv = OutConv(2 * number_of_classes, number_of_classes)

    def forward(self, x1, x2):
        # We process Sentinel1 input
        x1 = self.res_net_1(x1)
        # We process Sentinel2 input
        x2 = self.res_net_2(x2)

        # We concatenate the two representations
        x = torch.cat([x1, x2], dim=1)

        # We feed the fused representation to the output convolution
        x = self.out_conv(x)

        return x


# Create the last convolution block, responsible of the pixel classification
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
