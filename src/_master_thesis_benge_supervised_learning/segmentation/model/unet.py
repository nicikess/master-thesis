from torch import nn
from master_thesis_benge_supervised_learning.segmentation.model.unet_blocks import Down
from master_thesis_benge_supervised_learning.segmentation.model.unet_blocks import Up
from master_thesis_benge_supervised_learning.segmentation.model.unet_blocks import (
    DoubleConv,
)
from master_thesis_benge_supervised_learning.segmentation.model.unet_blocks import (
    OutConv,
)


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()

        self.inc = DoubleConv(in_channels, 64)

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

        # Last convolution block
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)

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

        x = self.outc(x)
        return x
