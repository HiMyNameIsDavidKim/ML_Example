import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# 이미지 세그멘테이션 모델 정의
class SimpleSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleSegmentationModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 데이터 로딩 및 데이터 전처리
train_dataset = datasets.Cityscapes(root='./data/cityscapes', split='train', mode='fine', target_type='semantic')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 모델 초기화 및 손실 함수 및 최적화 함수 설정
model = SimpleSegmentationModel(num_classes=19)  # Cityscapes dataset has 19 classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        break
    print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item()}')

print('학습 완료')
