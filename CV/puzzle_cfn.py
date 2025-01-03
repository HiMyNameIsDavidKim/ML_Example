import torch
import torch.nn as nn
import torchprofile
from torchsummary import summary
import time

# --------------------------------------------------------
# Ref : https://github.com/bbrattoli/JigsawPuzzlePytorch/blob/master/JigsawNetwork.py
# --------------------------------------------------------
# PuzzleCFN
# img_size=225, patch_size=75, num_puzzle=9
# input = [batch, 9, 3, 75, 75]
# FC1 : AlexNet, [batch, 9, 3, 75, 75] -> [batch, 9, 96, 33, 33]
# Pool : [batch, 9, 96, 16, 16]
# FC2 : AlexNet, [batch, 9, 96, 16, 16] -> [batch, 9, 256, 16, 16]
# Pool : [batch, 9, 256, 7, 7]
# FC3 : AlexNet, [batch, 9, 256, 7, 7] -> [batch, 9, 384, 7, 7]
# FC4 : AlexNet, [batch, 9, 384, 7, 7] -> [batch, 9, 384, 7, 7]
# FC5 : AlexNet, [batch, 9, 384, 7, 7] -> [batch, 9, 256, 7, 7]
# Pool : [batch, 9, 256, 3, 3]
# FC6 : [batch, 9, 256 * 3 * 3] -> [batch, 9, 1024]
# FC7 : [batch, 9 * 1024] -> [batch, 4096]
# FC8 : [batch, 4096] -> [batch, 1000]
# 1000개 클래스는 길이 9의 리스트로 각 요소는 1~9번 퍼즐의 순서 (ex. [9, 4, 6, 8, 3, 2, 5, 1, 7], 1번 퍼즐 위치는 9)
# --------------------------------------------------------


class PuzzleCFN(nn.Module):
    def __init__(self, classes=1000):
        super(PuzzleCFN, self).__init__()
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

        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1', nn.Linear(256 * 3 * 3, 1024))
        self.fc6.add_module('relu6_s1', nn.ReLU(inplace=True))
        self.fc6.add_module('drop6_s1', nn.Dropout(p=0.5))

        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7', nn.Linear(9 * 1024, 4096))
        self.fc7.add_module('relu7', nn.ReLU(inplace=True))
        self.fc7.add_module('drop7', nn.Dropout(p=0.5))

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc8', nn.Linear(4096, classes))

        # self.apply(weights_init)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.transpose(0, 1)

        x_list = []
        for i in range(9):
            z = self.conv(x[i])
            z = self.fc6(z.view(B, -1))
            z = z.view([B, 1, -1])
            x_list.append(z)

        x = torch.cat(x_list, 1)
        x = self.fc7(x.view(B, -1))
        x = self.classifier(x)

        return x


def weights_init(model):
    if type(model) in [nn.Conv2d, nn.Linear]:
        nn.init.xavier_normal(model.weight.data)
        nn.init.constant(model.bias.data, 0.1)

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


if __name__ == '__main__':
    model = PuzzleCFN(classes=1000)
    output = model(torch.rand(2, 9, 3, 75, 75))
    print(output.shape)
    summary(model, (9, 3, 75, 75))
