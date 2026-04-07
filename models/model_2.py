import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDC(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetDC, self).__init__()
        
        # Encoder (with progressively larger dilation factors)
        self.enc1 = self.double_conv(in_channels, 64, dilation=1)
        self.enc2 = self.double_conv(64, 128, dilation=2)
        self.enc3 = self.double_conv(128, 256, dilation=4)
        self.enc4 = self.double_conv(256, 512, dilation=8)

        # Bottleneck (large dilation)
        self.bottleneck = self.double_conv(512, 1024, dilation=16)

        # Decoder (commonly, we reduce or reset dilation in the decoder, 
        # but you can also experiment with continuing dilation here)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.double_conv(1024, 512, dilation=1)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.double_conv(512, 256, dilation=1)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.double_conv(256, 128, dilation=1)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.double_conv(128, 64, dilation=1)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels, dilation=1):
        """
        A block of two convolutions each followed by BatchNorm and ReLU,
        with adjustable dilation. We also adjust the padding to maintain
        output size properly: padding = dilation.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=dilation, dilation=dilation
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=dilation, dilation=dilation
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # --- Encoder ---
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # --- Bottleneck ---
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # --- Decoder ---
        up4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))

        up3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))

        up2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        out = self.out_conv(dec1)
        return torch.sigmoid(out)
