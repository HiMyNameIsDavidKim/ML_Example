import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
from torchsummary import summary


class FCViT(nn.Module):
    def __init__(self, num_puzzle=9, size_puzzle=75):
        super(FCViT, self).__init__()
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

    def random_shuffle(self, x):
        N, C, H, W = x.shape
        p = self.size_puzzle
        n = int(math.sqrt(self.num_puzzle))

        noise = torch.rand(N, self.num_puzzle, device=x.device)
        ids_shuffles = torch.argsort(noise, dim=1)
        ids_restores = torch.argsort(ids_shuffles, dim=1)

        for i, (img, ids_shuffle) in enumerate(zip(x, ids_shuffles)):
            fragments = [img[:, i:i + p, j:j + p] for i in range(0, H, p) for j in range(0, W, p)]
            shuffled_fragments = [fragments[idx] for idx in ids_shuffle]
            shuffled_fragments = [self.augment_fragment(piece) for piece in shuffled_fragments]
            shuffled_img = [torch.cat(row, dim=2) for row in [shuffled_fragments[i:i+n] for i in range(0, len(shuffled_fragments), n)]]
            shuffled_img = torch.cat(shuffled_img, dim=1)
            x[i] = shuffled_img

        start, end = 0, n
        self.min_dist = (end-start)/n
        self.map_values = list(torch.arange(start, end, self.min_dist))
        self.map_coord = torch.tensor([(i, j) for i in self.map_values for j in self.map_values])

        coord_shuffles = torch.zeros([N, self.num_puzzle, 2])
        coord_restores = torch.zeros([N, self.num_puzzle, 2])
        for i, (ids_shuffle, ids_restore) in enumerate(zip(ids_shuffles, ids_restores)):
            coord_shuffles[i] = self.map_coord[ids_shuffle]
            coord_restores[i] = self.map_coord[ids_restore]

        return x, coord_restores.to(x.device)

    def random_shuffle_1000(self, x):
        N, C, H, W = x.shape
        p = self.size_puzzle
        n = int(math.sqrt(self.num_puzzle))

        perm = np.load(f'./data/permutations_1000.npy')
        if np.min(perm) == 1:
            perm -= 1
        orders = [np.random.randint(len(perm)) for _ in range(N)]
        ids_shuffles = np.array([perm[o] for o in orders])
        ids_shuffles = torch.tensor(ids_shuffles, device=x.device)
        ids_restores = torch.argsort(ids_shuffles, dim=1)

        for i, (img, ids_shuffle) in enumerate(zip(x, ids_shuffles)):
            fragments = [img[:, i:i + p, j:j + p] for i in range(0, H, p) for j in range(0, W, p)]
            shuffled_fragments = [fragments[idx] for idx in ids_shuffle]
            shuffled_fragments = [self.augment_fragment(piece) for piece in shuffled_fragments]
            shuffled_img = [torch.cat(row, dim=2) for row in [shuffled_fragments[i:i+n] for i in range(0, len(shuffled_fragments), n)]]
            shuffled_img = torch.cat(shuffled_img, dim=1)
            x[i] = shuffled_img

        start, end = 0, n
        self.min_dist = (end-start)/n
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
        x, target = self.random_shuffle(x)
        x = x[:, :, :-1, :-1]

        x = self.vit_features(x)  # fc0 is included

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, self.num_puzzle, 2)

        return x, target


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


if __name__ == '__main__':
    model = FCViT()
    output, target = model(torch.rand(2, 3, 225, 225))
    summary(model, (3, 225, 225))
