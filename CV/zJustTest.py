import torch


class rgb2pentile(torch.nn.Module):
    def __init__(self):
        super().__init__()

        kernel_size = (3, 3)
        padding = 1
        channels = 256
        relu_inplace = True

        self.layers_f = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, padding=padding,
                            padding_mode='replicate'),
            torch.nn.ReLU(inplace=relu_inplace),

            torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding,
                            padding_mode='replicate'),
            torch.nn.ReLU(inplace=relu_inplace),

            torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding,
                            padding_mode='replicate'),
            torch.nn.ReLU(inplace=relu_inplace)
        )
        self.layers_b = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding,
                            padding_mode='replicate'),
            torch.nn.ReLU(inplace=relu_inplace),

            torch.nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=kernel_size, padding=padding,
                            padding_mode='replicate')
        )

    def forward(self, x, pooling_size=(2, 2)):
        x = self.layers_f(x)
        x = torch.nn.MaxPool2d(kernel_size=pooling_size)(x)
        x = self.layers_b(x)

        return x















import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

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

train_dataset = datasets.Cityscapes(root='./data/cityscapes', split='train', mode='fine', target_type='semantic')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = SimpleSegmentationModel(num_classes=19)  # Cityscapes dataset has 19 classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
