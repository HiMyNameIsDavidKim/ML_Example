import torch
import torch.nn as nn
from torchsummary import summary

class AlexNet(nn.Module):
    def __init__(self, classes=1000, **kwargs):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv1_s1', nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        self.conv.add_module('relu1_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool1_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn1_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv2_s1', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))
        self.conv.add_module('relu2_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool2_s1', nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv.add_module('lrn2_s1', LRN(local_size=5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv3_s1', nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv.add_module('relu3_s1', nn.ReLU(inplace=True))

        self.conv.add_module('conv4_s1', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('relu4_s1', nn.ReLU(inplace=True))

        self.conv.add_module('conv5_s1', nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))
        self.conv.add_module('relu5_s1', nn.ReLU(inplace=True))
        self.conv.add_module('pool5_s1', nn.MaxPool2d(kernel_size=3, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1), stride=1,padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size, stride=1,padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


def alex_base(**kwargs):
    model = AlexNet(classes=1000, **kwargs)
    return model


if __name__ == '__main__':
    model = AlexNet(classes=1000)
    output = model(torch.rand(2, 3, 224, 224))
    print(output.shape)
    summary(model, (3, 224, 224))

