import os

import torch
import pandas as pd

n_patches = 3
gap = 48
patch_size = 100
pieces = []
for j in range(n_patches):
    for k in range(n_patches):
        left = j * (patch_size + gap)
        upper = k * (patch_size + gap)
        right = left + patch_size
        lower = upper + patch_size
        print(left, right)
        print(upper, lower)
