from random import randint, randrange

import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

EPOCH = 10
BATCH_SIZE = 64
lr = 0.01
DEVICE = 'mps'

train = datasets.FashionMNIST(
    root='./data/',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
train_loader = DataLoader(
    dataset=train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)


class AEModel(nn.Module):
    def __init__(self):
        super(AEModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AutoEncoderModel(object):
    def __init__(self):
        self.train = None
        self.train_loader = None

    def process(self):
        self.train_model()

    def add_noise(self, img):
        noise = torch.randn(img.size()) * 0.2
        noisy_img = img.to(DEVICE) + noise.to(DEVICE)
        return noisy_img

    def train_model(self):
        model = AEModel().to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        view_data = train.data[:5].view(-1, 28 * 28)
        view_data = view_data.type(torch.FloatTensor) / 255.

        def closure(model, train_loader):
            model.train()
            avg_loss = 0
            for step, (x, label) in enumerate(train_loader):
                noisy_x = self.add_noise(x)
                noisy_x = noisy_x.view(-1, 28 * 28).to(DEVICE)
                y = x.view(-1, 28 * 28).to(DEVICE)

                label = label.to(DEVICE)
                encoded, decoded = model(noisy_x)

                loss = criterion(decoded, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
            return avg_loss / len(train_loader)

        for epoch in range(1, EPOCH + 1):
            loss = closure(model, train_loader)

            print(f"[Epoch {epoch}] loss: {loss}")

            original_x = view_data.to(DEVICE)
            noisy_x = self.add_noise(original_x).to(DEVICE)
            _, recovered_x = model(noisy_x)

            f, a = plt.subplots(3, 5, figsize=(5, 4))

            for i in range(5):
                original_img = np.reshape(original_x.to("cpu").data.numpy()[i], (28, 28))
                noisy_img = np.reshape(noisy_x.to("cpu").data.numpy()[i], (28, 28))
                recovered_img = np.reshape(recovered_x.to("cpu").data.numpy()[i], (28, 28))

                a[0][i].imshow(original_img, cmap='gray')
                a[0][i].axis("off")
                a[1][i].imshow(noisy_img, cmap='gray')
                a[1][i].axis("off")
                a[2][i].imshow(recovered_img, cmap='gray')
                a[2][i].axis("off")

            a[0, 2].set_title('Original image')
            a[1, 2].set_title('Noise image')
            a[2, 2].set_title('Encoding image')
            f.suptitle(f'[Epoch {epoch}]')
            plt.show()


if __name__ == '__main__':
    AutoEncoderModel().process()