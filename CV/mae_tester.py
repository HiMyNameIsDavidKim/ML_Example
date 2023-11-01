import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data as data
import torchvision
from torch.utils.data import random_split
from tqdm import tqdm, tqdm_notebook
import torch.nn.functional as F
import math
from functools import partial

import facebook_vit
from mae_util import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from CV.facebook_mae import MaskedAutoencoderViT

gpu_ids = []
device_names = []
if torch.cuda.is_available():
    for gpu_id in range(torch.cuda.device_count()):
        gpu_ids += [gpu_id]
        device_names += [torch.cuda.get_device_name(gpu_id)]
print(gpu_ids)
print(device_names)

if len(gpu_ids) > 1:
    gpu = 'cuda:' + str(gpu_ids[2])  # GPU Number
else:
    gpu = "cuda" if torch.cuda.is_available() else "cpu"


device = gpu
BATCH_SIZE = 512  # 1024
NUM_EPOCHS = 100  # 100
WARMUP_EPOCHS = 5  # 5
NUM_WORKERS = 2
LEARNING_RATE = 6.25e-05  # 1e-03
model_path = './save/mae_vit_base_i2012_ep100_lr6.25e-05.pt'


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
transform_test = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_set = torchvision.datasets.ImageFolder('../../YJ/ILSVRC2012/train', transform=transform_train)
train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size
train_set, val_set = random_split(train_set, [train_size, val_size])
train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_set = torchvision.datasets.ImageFolder('../../YJ/ILSVRC2012/val', transform=transform_test)
test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


class TesterFacebook(object):
    def __init__(self):
        self.model = None
        self.epochs = [0]
        self.losses = [0]
        self.accuracies = [0]

    def process(self):
        self.build_model()
        self.eval_model()

    def build_model(self):
        self.model = facebook_vit.__dict__['vit_base_patch16'](
            num_classes=1000,
            drop_path_rate=0.1,
        )
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(device)
        self.epochs = checkpoint['epochs']
        self.losses = checkpoint['losses']
        self.accuracies = checkpoint['accuracies']
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        print(f'Epochs: {self.epochs[-1]}')

    def eval_model(self):
        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in tqdm_notebook(enumerate(test_loader, 0), total=len(test_loader)):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc_test = 100 * correct / total
        print(f'Accuracy of {len(test_set)} test images: {acc_test:.2f} %')

        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in tqdm_notebook(enumerate(val_loader, 0), total=len(val_loader)):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc_val = 100 * correct / total
        print(f'Accuracy of {len(val_set)} val images: {acc_val:.2f} %')

        print(f'Accuracy of test: {acc_test:.2f} %, Accuracy of val: {acc_val:.2f} %')


if __name__ == '__main__':
    t = TesterFacebook()
    [t.process() for i in range(1)]