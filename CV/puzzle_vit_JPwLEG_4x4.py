import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
from torchsummary import summary

from util.tester import visualDoubleLoss


class JCViT(nn.Module):
    def __init__(self, num_puzzle=16, size_puzzle=56, threshold=0.8):
        super(JCViT, self).__init__()
        self.num_puzzle = num_puzzle
        self.size_puzzle = size_puzzle
        self.threshold = threshold
        self.vit_features = timm.create_model('vit_base_patch16_224', pretrained=False)
        # self.vit_features.head = nn.Linear(768, 1000)
        self.fc1 = nn.Linear(1000, 1000)
        self.fc2 = nn.Linear(1000, self.num_puzzle * 2)
        self.map_values = []
        self.map_coord = None
        self.min_dist = 0
        self.augment_tile = transforms.Compose([
            transforms.RandomCrop(72),
            transforms.Resize(56),
            transforms.Lambda(rgb_jittering),
            transforms.Lambda(tile_norm),
        ])

    def random_shuffle_JPwLEG(self, x):
        N, C, H, W = x.shape
        p = self.size_puzzle
        n = int(math.sqrt(self.num_puzzle))

        noise = torch.rand(N, self.num_puzzle, device=x.device)
        ids_shuffles = torch.argsort(noise, dim=1)
        ids_restores = torch.argsort(ids_shuffles, dim=1)

        x_ = torch.zeros(N, 3, 224, 224, device=x.device)
        for i, (img, ids_shuffle) in enumerate(zip(x, ids_shuffles)):
            # cropped_img = torch.zeros(3, 96 * 3, 96 * 3)
            n_patches = 4
            gap = 24
            patch_size = 74
            pieces = []
            for j in range(n_patches):
                for k in range(n_patches):
                    left = j * (patch_size + gap)
                    upper = k * (patch_size + gap)
                    right = left + patch_size
                    lower = upper + patch_size

                    patch = self.augment_tile(img[:, left:right, upper:lower])
                    pieces.append(patch)
                    # cropped_img[:, j * 96:j * 96 + 96, k * 96:k * 96 + 96] = patch
            shuffled_pieces = [pieces[idx] for idx in ids_shuffle]
            shuffled_img = [torch.cat(row, dim=2) for row in
                            [shuffled_pieces[i:i + n] for i in range(0, len(shuffled_pieces), n)]]
            shuffled_img = torch.cat(shuffled_img, dim=1)
            x_[i] = shuffled_img
        x = x_

        start, end = 0, n
        self.min_dist = (end - start) / n
        self.map_values = list(torch.arange(start, end, self.min_dist))
        self.map_coord = torch.tensor([(i, j) for i in self.map_values for j in self.map_values])

        coord_shuffles = torch.zeros([N, self.num_puzzle, 2])
        coord_restores = torch.zeros([N, self.num_puzzle, 2])
        for i, (ids_shuffle, ids_restore) in enumerate(zip(ids_shuffles, ids_restores)):
            coord_shuffles[i] = self.map_coord[ids_shuffle]
            coord_restores[i] = self.map_coord[ids_restore]

        return x, coord_restores.to(x.device)

    def mapping(self, target):
        diff = torch.abs(target.unsqueeze(3) - torch.tensor(self.map_values, device=target.device))
        min_indices = torch.argmin(diff, dim=3)
        target[:] = min_indices
        return target

    def forward(self, x):
        x, target = self.random_shuffle_JPwLEG(x)

        x = self.vit_features(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, self.num_puzzle, 2)

        return x, target


def rgb_jittering(tile):
    jitter_values = torch.randint(-2, 3, (3, 1, 1)).to(tile.device)
    jittered_tile = tile + jitter_values
    jittered_tile = torch.clamp(jittered_tile, 0, 255)
    return jittered_tile


def tile_norm(tile):
    m, s = torch.mean(tile.view(3, -1), dim=1).to(tile.device), torch.std(tile.view(3, -1), dim=1).to(tile.device)
    s[s == 0] = 1
    norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
    tile = norm(tile)
    return tile


if __name__ == '__main__':
    model = JCViT()
    output, target = model(torch.rand(2, 3, 468, 468))
    summary(model, (3, 468, 468))
