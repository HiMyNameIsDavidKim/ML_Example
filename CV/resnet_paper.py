import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import pickle

device = 'mps'
path = './save/resnet_paper.pt'
batch_size = 128
lr = [0.1, 0.01, 0.001]
num_epoch = 185

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = dset.CIFAR10(root='./data/',
                         train=True,
                         download=True,
                         transform=transform)
test_set = dset.CIFAR10(root='./data/',
                        train=False,
                        download=True,
                        transform=transform)
train_ds, val_ds = random_split(train_set, [45000, 5000])
test_ds = test_set
train_loader = DataLoader(train_ds,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)
val_loader = DataLoader(val_ds,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=2)
test_loader = DataLoader(test_ds,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=2)

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


class BasicBlock(nn.Module):
    expand = 1

    def __init__(self, in_size, out_size, stride=1):
        super(BasicBlock, self).__init__()
        self.cnn1 = nn.Conv2d(in_size, out_size, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.cnn2 = nn.Conv2d(out_size, out_size, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_size)

        self.identity = nn.Sequential()
        if stride != 1:
            self.identity = nn.Sequential(
                nn.Conv2d(in_size, out_size, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        i = self.identity(x)
        x = self.bn1(self.cnn1(x))
        x = F.relu(x)
        x = self.bn2(self.cnn2(x))
        x = F.relu(x + i)
        return x


class BottleNeckBlock(nn.Module):
    expand = 4

    def __init__(self, in_size, out_size, stride=1):
        super(BottleNeckBlock, self).__init__()
        self.cnn1 = nn.Conv2d(in_size, out_size, 1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.cnn2 = nn.Conv2d(out_size, out_size, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.cnn3 = nn.Conv2d(out_size, out_size * self.expand, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_size * self.expand)

        self.identity = nn.Sequential()
        if stride != 1 or in_size != out_size * self.expand:
            self.identity = nn.Sequential(
                nn.Conv2d(in_size, out_size * self.expand, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_size * self.expand)
            )

    def forward(self, x):
        i = self.identity(x)
        x = self.bn1(self.cnn1(x))
        x = F.relu(x)
        x = self.bn2(self.cnn2(x))
        x = F.relu(x)
        x = self.bn3(self.cnn3(x))
        x = F.relu(x + i)
        return x


class ResnetPaperModel(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResnetPaperModel, self).__init__()
        self.in_size = 16
        self.cnn1 = nn.Conv2d(3, self.in_size, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_size)
        self.cnn2 = self.make_layer(BasicBlock, 16, num_blocks[0], stride=1)
        self.cnn3 = self.make_layer(BasicBlock, 32, num_blocks[1], stride=2)
        self.cnn4 = self.make_layer(BasicBlock, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64 * block.expand, 10)

    def make_layer(self, block, out_size, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_size, out_size, stride=stride))
            self.in_size = out_size * block.expand
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn1(self.cnn1(x))
        x = F.relu(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ClassifyModel(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.errors = []

    def process(self):
        self.pre_ds()
        self.train_model()

    def pre_ds(self):
        print(len(train_set), len(test_set))
        print(len(train_ds), len(val_ds), len(test_ds))

    def train_model(self):
        model = ResnetPaperModel(BasicBlock, [3, 3, 3]).to(device)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr[0])

        iters = 0
        loss_arr = []
        for epoch in range(num_epoch):
            for j, [image, label] in enumerate(train_loader):
                x = image.to(device)
                y_ = label.to(device)

                optimizer.zero_grad()
                output = model.forward(x)
                loss = loss_func(output, y_)
                loss.backward()
                optimizer.step()

                if iters % 1000 == 0:
                    print(f'[iters: {iters}] ({loss:.4f}, device is {x.device})')
                    loss_arr.append(loss.cpu().detach().numpy())
                    self.save_model(iters, model, optimizer, loss)
                    self.load_model()
                    self.eval_model()
                    self.error_graph()
                iters += 1

                if iters == 32000:
                    optimizer = optim.Adam(model.parameters(), lr=lr[1])
                elif iters == 48000:
                    optimizer = optim.Adam(model.parameters(), lr=lr[2])

    def error_graph(self):
        error_arr = self.errors
        plt.figure(figsize=(5, 5))
        plt.xlabel('Iteration (1e3)')
        plt.ylabel('Error (%)')
        plt.plot(error_arr)
        plt.show()

    def save_model(self, iters, model, optimizer, loss):
        torch.save({
            'iters': iters,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)

    def load_model(self):
        model = ResnetPaperModel(BasicBlock, [3, 3, 3])

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        iters = checkpoint['iters']
        loss = checkpoint['loss']

        if iters < 32000:
            optimizer = optim.Adam(model.parameters(), lr=lr[0])
        elif iters < 48000:
            optimizer = optim.Adam(model.parameters(), lr=lr[1])
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr[2])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.model = model
        self.optimizer = optimizer

    def eval_model(self):
        model = self.model.to(device)
        correct = 0
        total = 0

        model.eval()
        with torch.no_grad():
            for image, label in val_loader:
                x = image.to(device)
                y_ = label.to(device)

                output = model.forward(x)
                _, output_index = torch.max(output, 1)

                total += label.size(0)
                correct += (output_index == y_).sum().float()

            error = 100 - (100 * correct / total)
            self.errors.append(error.cpu().detach().numpy())
            print(f"Error of Validation Data: {error:.2f}%")

    def test_model(self):
        model = self.model.to(device)
        correct = 0
        total = 0

        model.eval()
        with torch.no_grad():
            for image, label in test_loader:
                x = image.to(device)
                y_ = label.to(device)

                output = model.forward(x)
                _, output_index = torch.max(output, 1)

                total += label.size(0)
                correct += (output_index == y_).sum().float()

            error = 100 - (100 * correct / total)
            self.errors.append(error.cpu().detach().numpy())
            print(f"Error of Validation Data: {error:.2f}%")


if __name__ == '__main__':
    # resnet = ResnetPaperModel(BasicBlock, [3, 3, 3])
    # summary(resnet, (3, 32, 32), batch_size=1)
    # ClassifyModel().process()
    cm = ClassifyModel()
    cm.load_model()
    cm.test_model()
