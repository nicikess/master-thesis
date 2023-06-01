from torch import nn
import torch
import torch.nn.functional as F

class TripleUNet(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, in_channels_3, number_of_classes):
        super(TripleUNet, self).__init__()
        
        # First stream of UNet()
        self.unet1 = UNet(in_channels_1, number_of_classes=number_of_classes)
        # Second stream of UNet()
        self.unet2 = UNet(in_channels_2, number_of_classes=number_of_classes)
        # Third stream of UNet()
        self.unet3 = UNet(in_channels_3, number_of_classes=number_of_classes)
        
        # Output convolution
        self.outc = OutConv(3 * 64, number_of_classes)

    def forward(self, x1, x2, x3):
        # We process Sentinel1 input
        x1 = self.unet1(x1)
        # We process Sentinel2 input
        x2 = self.unet2(x2)

        x3 = self.unet3(x3)

        '''
        Both unet output a tensor of shape [32,64,120,120]
        Then those are concatenated along axis 1 to a tensor of shape [32,128,120,120]
        Then the output convolution of shape [32,11,120,120] is caluclated
        '''
        
        # We concatenate the two representations
        x = torch.cat([x1, x2, x3], dim=1)
        
        # We feed the fused representation to the output convolution
        x = self.outc(x)
        
        return x


class UNet(nn.Module):
    def __init__(self, in_channels_1, number_of_classes):
        super(UNet, self).__init__()

        self.inc = DoubleConv(in_channels_1, 64)

        # Initialise the Encoder
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // 2)

        # Initialise the Decoder
        self.up1 = Up(1024, 512 // 2)
        self.up2 = Up(512, 256 // 2)
        self.up3 = Up(256, 128 // 2)
        self.up4 = Up(128, 64)

    def forward(self, x1):
        x1 = self.inc(x1)

        # In these 4 downsampling blocks, the size of the image is gradually reduced
        # while the depth is gradually increased.
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # In these 4 upsampling  blocks, the size of the image is gradually increased
        # while the depth is gradually reduced.
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return x


# Create the convolution block
class DoubleConv(nn.Module):
    """
    The DoubleConv object is composed of two successive blocks of convolutional layers, batch normalization and ReLU.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # Create a sequential module.
        # nn.Sequential is a module inside which you can put other modules that will be applied one after the other.
        self.double_conv = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # First batchnormalization
            nn.BatchNorm2d(mid_channels),
            # First ReLU activation function
            nn.ReLU(inplace=True),
            # Second convolutional layer
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # Second batchnormalization
            nn.BatchNorm2d(out_channels),
            # Second ReLU activation function
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


# Create the downsampling block
class Down(nn.Module):
    """
    The Down object is composed of a maxpooling layer followed by the DoubleConv block defined above.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Create a sequential module.
        self.maxpool_conv = nn.Sequential(
            # 2D max pooling layer with a kernel size of 2 (meaning spatial dimension will be divided by two)
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# Create the upsampling block
class Up(nn.Module):
    """
    The Up object is composed of an upsampling layer (bilinear interpolation) followed by the DoubleConv block defined above.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # upsampling layer with a scale factor of 2 (meaning spatial dimension will be multiplied by two)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# Create the last convolution block, responsible of the pixel supervised_baseline
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
