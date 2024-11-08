import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
from torchsummary import summary


class FCGen(nn.Module):
    def __init__(self, num_puzzle=9, size_puzzle=75):
        super(FCGen, self).__init__()
        self.num_puzzle = num_puzzle
        self.size_puzzle = size_puzzle
        self.vit_features = timm.create_model('vit_base_patch16_224', pretrained=False)
        # self.vit_features.head = nn.Linear(768, 1000)  # fc0
        self.fc1 = nn.Linear(1000, 1000)
        self.fc2 = nn.Linear(1000, self.num_puzzle * 2)
        self.map_values = []
        self.map_coord = None
        self.augment_fragment = transforms.Compose([
            transforms.RandomCrop((64, 64)),
            transforms.Resize((75, 75)),
            transforms.Lambda(rgb_jittering),
            transforms.Lambda(fragment_norm),
        ])

    def augmentation(self, x):
        N, C, H, W = x.shape
        p = self.size_puzzle
        n = int(math.sqrt(self.num_puzzle))

        origin_order = torch.arange(9, device=x.device).repeat(N, 1)

        for i, (img, ids) in enumerate(zip(x, origin_order)):
            fragments = [img[:, i:i + p, j:j + p] for i in range(0, H, p) for j in range(0, W, p)]
            fragments = [self.augment_fragment(piece) for piece in fragments]
            augmented_img = [torch.cat(row, dim=2) for row in [fragments[i:i+n] for i in range(0, len(fragments), n)]]
            augmented_img = torch.cat(augmented_img, dim=1)
            x[i] = augmented_img

        start, end = 0, n
        self.min_dist = (end-start)/n
        self.map_values = list(torch.arange(start, end, self.min_dist))
        self.map_coord = torch.tensor([(i, j) for i in self.map_values for j in self.map_values])

        return x

    def mapping2perm(self, x):
        diff = torch.abs(x.unsqueeze(3) - torch.tensor(self.map_values, device=x.device))
        min_indices = torch.argmin(diff, dim=3)
        x[:] = min_indices
        x = self.coord2perm(x)
        return x

    def coord2perm(self, x):
        N, h, v = x.shape
        x_perm = torch.empty((N, h), device=x.device)
        dic = {
            (0, 0): 0, (0, 1): 1, (0, 2): 2,
            (1, 0): 3, (1, 1): 4, (1, 2): 5,
            (2, 0): 6, (2, 1): 7, (2, 2): 8
        }

        for i, b in enumerate(x):
            for j, x_coord in enumerate(b):
                x_perm[i, j] = dic[tuple(x_coord.tolist())]
        return x_perm

    def forward(self, x):
        x = self.augmentation(x)
        x = x[:, :, :-1, :-1]

        x = self.vit_features(x)  # fc0 is included

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, self.num_puzzle, 2)
        x = self.mapping2perm(x)

        return x


def rgb_jittering(fragment):
    jitter_values = torch.randint(-2, 3, (3, 1, 1)).to(fragment.device)
    jittered_fragment = fragment + jitter_values
    jittered_fragment = torch.clamp(jittered_fragment, 0, 255)
    return jittered_fragment


def fragment_norm(fragment):
    m, s = torch.mean(fragment.view(3, -1), dim=1).to(fragment.device), torch.std(fragment.view(3, -1), dim=1).to(fragment.device)
    s[s == 0] = 1
    norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
    fragment = norm(fragment)
    return fragment


def inverse_loss(outputs, targets, beta=1.0, reduction='mean'):
    diff = torch.abs(outputs - targets)
    loss = torch.where(diff < beta, 2 - 0.5*(diff**2), 1.5 - diff)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


if __name__ == '__main__':
    model = FCGen()
    output, target = model(torch.rand(2, 3, 225, 225))
    summary(model, (3, 225, 225))
