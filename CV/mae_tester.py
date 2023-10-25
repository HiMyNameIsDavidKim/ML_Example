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

from facebook_mae import MaskedAutoencoderViT

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
model_path = './data/mae_checkpoint/mae_finetuned_vit_base.pth'
BATCH_SIZE = 512
NUM_EPOCHS = 8
NUM_WORKERS = 2
LEARNING_RATE = 1.25e-03


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
train_set = torchvision.datasets.ImageFolder('./data/ImageNet/val', transform=transform_train)
train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size
train_set, val_set = random_split(train_set, [train_size, val_size])
train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_set = torchvision.datasets.ImageFolder('./data/ImageNet/val', transform=transform_test)
test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


class TesterTimm(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epochs = []
        self.losses = []

    def process(self):
        self.build_model()
        # self.eval_model()

    def build_model(self):
        self.model = MaskedAutoencoderViT(patch_size=16,
                                          embed_dim=768,
                                          depth=12,
                                          num_heads=12,
                                          decoder_embed_dim=512,
                                          decoder_depth=8,
                                          decoder_num_heads=16,
                                          mlp_ratio=4,
                                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                          ).to(device)
        checkpoint = torch.load(model_path)
        print(checkpoint.keys())
        # self.epochs = checkpoint['epochs']
        self.model.load_state_dict(checkpoint['model'])
        # self.losses = checkpoint['losses']
        # print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        # print(f'Classes: {self.model.num_classes}')
        # print(f'Epochs: {self.epochs[-1]}')

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
        print(f'Accuracy of {len(test_set)} test images: {100 * correct / total:.2f} %')

    def lr_checker(self):
        self.build_model()
        model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0
            saving_loss = 0.0
            print(optimizer.param_groups[0]['lr'])
            scheduler.step()


if __name__ == '__main__':
    t = TesterTimm()
    [t.process() for i in range(1)]