from itertools import permutations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math


device = 'cpu'
lr = 3e-04
batch_size = 64
num_epochs = 20


transform = transforms.Compose([
    transforms.CenterCrop(30),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class PuzzleCNN(nn.Module):
    def __init__(self, num_puzzle=9, size_puzzle=10):
        super(PuzzleCNN, self).__init__()
        self.num_puzzle = num_puzzle
        self.size_puzzle = size_puzzle
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, self.num_puzzle * self.num_puzzle)

    def random_shuffle(self, x):
        N, C, H, W = x.shape
        p = self.size_puzzle
        n = int(math.sqrt(self.num_puzzle))

        noise = torch.rand(N, self.num_puzzle, device=x.device)
        ids_shuffles = torch.argsort(noise, dim=1)
        ids_restores = torch.argsort(ids_shuffles, dim=1)

        for i, (img, ids_shuffle) in enumerate(zip(x, ids_shuffles)):
            pieces = [img[:, i:i + p, j:j + p] for i in range(0, H, p) for j in range(0, W, p)]
            shuffled_pieces = [pieces[idx] for idx in ids_shuffle]
            shuffled_img = [torch.cat(row, dim=2) for row in [shuffled_pieces[i:i+n] for i in range(0, len(shuffled_pieces), n)]]
            shuffled_img = torch.cat(shuffled_img, dim=1)
            x[i] = shuffled_img

        return x, ids_restores

    def forward(self, x):
        x, target = self.random_shuffle(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = x.view(-1, self.num_puzzle, self.num_puzzle)
        return x, target


# 모든 경우의 수 (0~15)의 순열을 클래스로 만들고 이게 매칭되게 도와줘야함.
# 이러면 갯수가 너무 많은데............


if __name__ == '__main__':
    list_ = list(range(9))
    list_perm = list(permutations(list_))
    print(len(list_perm))

    # model = PuzzleCNN().to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=lr)
    #
    # for epoch in range(num_epochs):
    #     # train
    #     model.train()
    #     for batch_idx, (inputs, _) in enumerate(train_loader):
    #         inputs = inputs.to(device)
    #
    #         outputs, labels = model(inputs)
    #
    #         optimizer.zero_grad()
    #
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         if batch_idx % 300 == 0:
    #             print(f'[Epoch {epoch}] [Batch {batch_idx}] Loss: {loss.item():.4f}')
    #
    #     # test
    #     model.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for inputs, _ in test_loader:
    #             inputs = inputs.to(device)
    #
    #             outputs, labels = model(inputs)
    #
    #             _, pred = torch.max(outputs.data, 1)
    #             total += labels.size(0) * labels.size(1)
    #             correct += (pred == labels).sum().item()
    #
    #     print(f'[Epoch {epoch}] Accuracy on the test set: {100 * correct / total:.2f}%')
