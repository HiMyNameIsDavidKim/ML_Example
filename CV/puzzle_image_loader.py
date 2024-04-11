import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from itertools import permutations

# --------------------------------------------------------
# Ref : https://github.com/bbrattoli/JigsawPuzzlePytorch/blob/master/Dataset/JigsawImageLoader.py
# --------------------------------------------------------
class PuzzleDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        numbers = list(range(9))
        permutation_list = permutations(numbers)
        permutations_array = np.array(list(permutation_list))
        self.permutations = permutations_array

    def __getitem__(self, index):
        img, label = self.dataset[index]

        C, H, W = img.shape
        p = int(H/3)

        pieces = [img[:, i:i + p, j:j + p] for i in range(0, H, p) for j in range(0, W, p)]
        tiles = pieces

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data = torch.stack(data, 0)

        return data, int(order), tiles

    def __len__(self):
        return len(self.dataset)


class PuzzleDataset1000(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.permutations = self.retrive_permutations(1000)
        self.augment_tile = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Resize((75, 75)),
            transforms.Lambda(rgb_jittering),
            transforms.Lambda(tile_norm),
        ])

    def __getitem__(self, index):
        img, label = self.dataset[index]

        C, H, W = img.shape
        p = int(H/3)

        pieces = [img[:, i:i + p, j:j + p] for i in range(0, H, p) for j in range(0, W, p)]
        tiles = pieces

        order = np.random.randint(len(self.permutations))
        data = [tiles[self.permutations[order][t]] for t in range(9)]
        data = [self.augment_tile(tile) for tile in data]
        data = torch.stack(data, 0)

        return data, int(order), tiles

    def __len__(self):
        return len(self.dataset)

    def retrive_permutations(self, classes):
        all_perm = np.load(f'./data/permutations_{classes}.npy')
        if all_perm.min() == 1:
            all_perm = all_perm - 1
        return all_perm


def rgb_jittering(tile):
    jitter_values = torch.randint(-2, 3, (3, 1, 1))
    jittered_tile = tile + jitter_values
    jittered_tile = torch.clamp(jittered_tile, 0, 255)
    return jittered_tile


def tile_norm(tile):
    m, s = tile.view(3, -1).mean(dim=1).numpy(), tile.view(3, -1).std(dim=1).numpy()
    s[s == 0] = 1
    norm = transforms.Normalize(mean=m.tolist(), std=s.tolist())
    tile = norm(tile)
    return tile
