from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from tqdm import tqdm

dataroot = "./data/celeba"
workers = 2
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
lr = 0.0002
beta1 = 0.5
ngpu = 1
device = 'mps'


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Celeba(object):
    def __init__(self):
        self.dataset = None
        self.dataloader = None
        self.img_list = []
        self.num_epoch = 0
        self.real_batch = None

    def process(self):
        self.select_data()
        self.training(int(input(f'How many epoch? : ')))
        self.show_result()

    def new_seed(self):
        manual_seed = random.randint(1, 10000)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        print(f'Seed : {manual_seed}')

    def select_data(self):
        self.dataset = dset.ImageFolder(root=dataroot,
                                        transform=transforms.Compose([
                                            transforms.Resize(image_size),
                                            transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ]))
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                      shuffle=True, num_workers=workers)
        self.show_dataset()

    def show_dataset(self):
        self.real_batch = next(iter(self.dataloader))

        plt.figure(figsize=(15, 15))
        plt.title("Real Images")
        plt.imshow(np.transpose(
            vutils.make_grid(self.real_batch[0].to('cpu')[:64], padding=5, normalize=True).to('cpu'), (1, 2, 0)))
        plt.show()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def training(self, ep):
        self.num_epoch = ep
        netG = Generator(ngpu).to(device)
        netD = Discriminator(ngpu).to(device)
        criterion = nn.BCELoss()
        fixed_noise = torch.randn(64, nz, 1, 1).to(device)
        real_label = 1
        fake_label = 0
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        for epoch in range(self.num_epoch):
            for i, data in enumerate(tqdm(self.dataloader)):

                netD.zero_grad()
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float).to(device)
                output = netD(real_cpu).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(b_size, nz, 1, 1).to(device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

                netG.zero_grad()
                label.fill_(real_label)
                output = netD(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                if (iters % 500 == 0) or ((epoch == self.num_epoch - 1) and (i == len(self.dataloader) - 1)):
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().to('cpu')
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                iters += 1
            print(f'[{epoch+1}/{self.num_epoch}]\t'
                  f'Loss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\t'
                  f'D(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            self.show_result(epoch+1)

    def show_result(self, num):
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(
            vutils.make_grid(self.real_batch[0].to('cpu')[:64], padding=5, normalize=True).to('cpu'), (1, 2, 0)))

        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(
            self.img_list[-1], (1, 2, 0)))
        plt.savefig(f'./save/celeba/celeba_{num}.jpg', dpi=300)
        plt.show()


celeb_menus = ["Exit",  # 0
               "Take New Seed",  # 1
               "Check Image",  # 2
               "Training",  # 3
               ]

celeb_lambda = {
    "1": lambda t: t.new_seed(),
    "2": lambda t: t.select_data(),
    "3": lambda t: t.training(int(input(f'How many epoch? : '))),
    "4": lambda t: print(" ** No Function ** "),
    "5": lambda t: print(" ** No Function ** "),
    "6": lambda t: print(" ** No Function ** "),
    "7": lambda t: print(" ** No Function ** "),
    "8": lambda t: print(" ** No Function ** "),
    "9": lambda t: print(" ** No Function ** "),
}

if __name__ == '__main__':
    c = Celeba()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(celeb_menus)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("### Exit ###")
            break
        else:
            try:
                celeb_lambda[menu](c)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")