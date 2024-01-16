import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

# 데이터셋 클래스 정의
class PuzzleDataset(Dataset):
    def __init__(self, mnist_dataset, transform):
        self.mnist_dataset = mnist_dataset
        self.transform = transform

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        image, _ = self.mnist_dataset[idx]
        image = self.transform(image)

        # 이미지를 4x4 퍼즐로 나누기
        puzzle = []
        for i in range(4):
            for j in range(4):
                puzzle.append(image[:, i * 7:(i + 1) * 7, j * 7:(j + 1) * 7])

        # 이미지 위치 레이블 생성 (0~15)
        label = torch.tensor([i for i in range(16)])

        return puzzle, label

# 신경망 모델 정의
class PuzzleSolver(nn.Module):
    def __init__(self):
        super(PuzzleSolver, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 16)

    def forward(self, x):
        x = torch.cat(x, dim=1)  # 각 퍼즐 조각을 하나의 텐서로 합치기
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


# 학습 파라미터 설정
device = 'mps'
batch_size = 64
learning_rate = 0.0001
epochs = 10

# 데이터 로드 및 전처리
transform = transforms.Compose([
    transforms.Resize((28, 28)),
])

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
puzzle_dataset = PuzzleDataset(mnist_dataset, tensor_transform)

dataloader = DataLoader(puzzle_dataset, batch_size=batch_size, shuffle=True)

# 모델, 손실 함수, 옵티마이저 초기화
model = PuzzleSolver()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
for epoch in range(epochs):
    for inputs, labels in dataloader:
        inputs, labels = inputs, labels
        labels = labels.float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(labels[0])
        print(outputs[0])

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')


