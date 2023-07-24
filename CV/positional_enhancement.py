import timm
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.optim import AdamW, SGD
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
print(f'Parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

