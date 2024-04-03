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


def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')
