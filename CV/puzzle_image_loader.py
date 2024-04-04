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

    def retrive_permutations(self, classes):
        all_perm = np.load(f'./save/permutations_{classes}.npy')
        if all_perm.min() == 1:
            all_perm = all_perm - 1
        return all_perm