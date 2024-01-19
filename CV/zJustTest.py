from itertools import permutations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
import math

from tqdm import tqdm

device = 'cpu'
lr = 3e-05
batch_size = 64
NUM_EPOCHS = 1


transform = transforms.Compose([
    transforms.Pad(padding=3),
    transforms.CenterCrop(30),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
train_loader.dataset.data = train_loader.dataset.data[:2000]
train_loader.dataset.targets = train_loader.dataset.targets[:2000]
test_loader.dataset.data = train_loader.dataset.data[:2000]
test_loader.dataset.targets = train_loader.dataset.targets[:2000]

class PuzzleCNNCoord(nn.Module):
    def __init__(self):
        super(PuzzleCNNCoord, self).__init__()
        resnet = resnet18(pretrained=True)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, 25 * 2)

    def forward(self, x):
        x = self.resnet_features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = x.view(-1, 25, 2)
        return x


if __name__ == '__main__':
    model = PuzzleCNNCoord().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    for epoch in range(NUM_EPOCHS):
        # train
        model.train()
        for batch_idx, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs = inputs.to(device)

            outputs = model(inputs)

            optimizer.zero_grad()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'[Epoch {epoch}] [Batch {batch_idx}] Loss: {loss.item():.4f}')

        scheduler.step()

        # test
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, total=len(test_loader)):
                inputs = inputs.to(device)

                outputs = model(inputs)

                outputs = F.softmax(outputs, dim=1)
                _, pred = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (pred == labels).sum().item()

        print(f'[Epoch {epoch}] Accuracy on the test set: {100 * correct / total:.2f}%')
