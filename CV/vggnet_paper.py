import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGGNet, self).__init__()
        self.cnn1 = nn.Conv2d(3, 64, 3, padding=1)
        self.cnn2 = nn.Conv2d(64, 64, 3, padding=1)
        self.cnn3 = nn.Conv2d(64, 128, 3, padding=1)
        self.cnn4 = nn.Conv2d(128, 128, 3, padding=1)
        self.cnn5 = nn.Conv2d(128, 256, 3, padding=1)
        self.cnn6 = nn.Conv2d(256, 256, 3, padding=1)
        self.cnn7 = nn.Conv2d(256, 512, 3, padding=1)
        self.cnn8 = nn.Conv2d(512, 512, 3, padding=1)

        self.maxPool1 = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = self.maxPool1(x)
        x = F.relu(self.cnn3(x))
        x = F.relu(self.cnn4(x))
        x = self.maxPool1(x)
        x = F.relu(self.cnn5(x))
        x = F.relu(self.cnn6(x))
        x = F.relu(self.cnn6(x))
        x = self.maxPool1(x)
        x = F.relu(self.cnn7(x))
        x = F.relu(self.cnn8(x))
        x = F.relu(self.cnn8(x))
        x = self.maxPool1(x)
        x = F.relu(self.cnn8(x))
        x = F.relu(self.cnn8(x))
        x = F.relu(self.cnn8(x))
        x = self.maxPool1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = self.fc3(x)
        return x