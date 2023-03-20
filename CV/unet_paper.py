import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder_conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv5 = DoubleConv(512, 1024)

        self.upconv6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_conv6 = DoubleConv(1024, 512)
        self.upconv7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_conv7 = DoubleConv(512, 256)
        self.upconv8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv8 = DoubleConv(256, 128)
        self.upconv9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv9 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder_conv1(x)
        x2 = self.encoder_conv2(self.pool1(x1))
        x3 = self.encoder_conv3(self.pool2(x2))
        x4 = self.encoder_conv4(self.pool3(x3))
        x5 = self.encoder_conv5(self.pool4(x4))

        x = self.upconv6(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.decoder_conv6(x)
        x = self.upconv7(x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder_conv7(x)
        x = self.upconv8(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder_conv8(x)
        x = self.upconv9(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder_conv9(x)

        x = self.final_conv(x)
        return x