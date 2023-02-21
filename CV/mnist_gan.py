import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import torchvision.utils as vutils
from torchvision.utils import save_image


num_epoch = 200
batch_size = 100
learning_rate = 0.0002
img_size = 28 * 28
num_channel = 1
noise_size = 100
hidden_size1 = 256
hidden_size2 = 512
hidden_size3 = 1024
save_path = f'./save/mnist_GAN.pt'
device = 'mps'


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(img_size, hidden_size2)
        self.linear2 = nn.Linear(hidden_size2, hidden_size1)
        self.linear3 = nn.Linear(hidden_size1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.linear1(x))
        x = self.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.linear1 = nn.Linear(noise_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, img_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.tanh(x)
        return x


class MNISTGan(object):
    def __init__(self):
        self.data_loader = None

    def process(self):
        self.create_data()
        self.train_model()

    def create_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)])
        dataset = dset.MNIST(root="./data/", train=True,
                             transform=transform,
                             target_transform=None,
                             download=True)
        self.data_loader = DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=True)

    def train_model(self):
        discriminator = Discriminator().to(device)
        generator = Generator().to(device)

        criterion = nn.BCELoss()
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

        for epoch in range(num_epoch):
            for i, (images, label) in enumerate(self.data_loader):
                real_label = torch.full((batch_size, 1), 1, dtype=torch.float32).to(device)
                fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32).to(device)
                real_images = images.reshape(batch_size, -1).to(device)

                g_optimizer.zero_grad()
                d_optimizer.zero_grad()
                z = torch.randn(batch_size, noise_size).to(device)
                fake_images = generator(z)
                g_loss = criterion(discriminator(fake_images), real_label)
                g_loss.backward()
                g_optimizer.step()

                d_optimizer.zero_grad()
                g_optimizer.zero_grad()
                z = torch.randn(batch_size, noise_size).to(device)
                fake_images = generator(z)
                fake_loss = criterion(discriminator(fake_images), fake_label)
                real_loss = criterion(discriminator(real_images), real_label)
                d_loss = (fake_loss + real_loss) / 2
                d_loss.backward()
                d_optimizer.step()

                d_performance = discriminator(real_images).mean()
                g_performance = discriminator(fake_images).mean()

                if (i + 1) % 600 == 0:
                    print(f"Epoch [ {epoch}/{num_epoch} ]  "
                          f"d_loss : {d_loss.item():.5f}  g_loss : {g_loss.item():.5f}")

            print(f"discriminator performance : {d_performance:.2f}  "
                  f"generator performance : {g_performance:.2f}")

            if epoch % 10 == 0:
                samples = fake_images.reshape(batch_size, 1, 28, 28)
                plt.figure(figsize=(10, 10))
                plt.axis("off")
                plt.imshow(vutils.make_grid(samples.to('cpu'), nrow=10).permute(1, 2, 0))
                plt.savefig(f'./save/mnist_gan/fake_samples_{epoch}.png', dpi=300)
                plt.show()

    # def save_model(self, model_a, model_b, opti_a, opti_b):
    #     torch.save({
    #         'modelA_state_dict': model_a.state_dict(),
    #         'modelB_state_dict': model_b.state_dict(),
    #         'optimizerA_state_dict': opti_a.state_dict(),
    #         'optimizerB_state_dict': opti_b.state_dict(),
    #     }, save_path)
    #
    # def load_model(self):
    #     discriminator = Discriminator().to(device)
    #     generator = Generator().to(device)
    #     d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    #     g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    #
    #     checkpoint = torch.load(save_path)
    #     discriminator.load_state_dict(checkpoint['modelA_state_dict'])
    #     generator.load_state_dict(checkpoint['modelB_state_dict'])
    #     d_optimizer.load_state_dict(checkpoint['optimizerA_state_dict'])
    #     g_optimizer.load_state_dict(checkpoint['optimizerB_state_dict'])
    #
    #     # eval or train again.


if __name__ == '__main__':
    MNISTGan().process()
