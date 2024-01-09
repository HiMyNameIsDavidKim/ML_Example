import torch
import torch.nn as nn
import torchvision.transforms as transforms
import PIL
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data as data
import torchvision
from torch.utils.data import random_split
from tqdm import tqdm, tqdm_notebook
import torch.nn.functional as F
import math
from functools import partial
import matplotlib.pyplot as plt

import facebook_vit
import facebook_mae
import shuffled_mae_case2
from mae_util import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from facebook_mae import MaskedAutoencoderViT

from util.tester import visualLossAcc, visualMultiLoss
from util.visualization import inout_images_plot, acc_jigsaw

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
BATCH_SIZE = 64  # 1024 // 64
NUM_EPOCHS = 800  # 100 // 800
WARMUP_EPOCHS = 40  # 5 // 40
NUM_WORKERS = 2
LEARNING_RATE = 1.5e-04  # paper: 1e-03 // 1.5e-04 -> implementation: 5e-04 // 1.5e-04
model_path = './save/mae/mae_vit_large_i2012_ep10_lr2e-06.pt'
given_model_path = './save/MAE/mae_visualize_vit_large_given.pth'


transform_train = transforms.Compose([
    transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_set = torchvision.datasets.ImageFolder('./data/ImageNet/val', transform=transform_train)
train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size
train_set, val_set = random_split(train_set, [train_size, val_size])
train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_set = torchvision.datasets.ImageFolder('./data/ImageNet/val', transform=transform_test)
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
        self.model = facebook_vit.__dict__['vit_large_patch16'](
            num_classes=1000,
            drop_path_rate=0.1,
            global_pool=True,
        )
        checkpoint = torch.load(model_path, map_location=device)
        checkpoint_model = checkpoint['model']
        self.model.load_state_dict(checkpoint_model)
        self.model.to(device)
        if 'given' not in str(model_path):
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

    def lr_checker(self):
        self.build_model()
        model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        for epoch in range(NUM_EPOCHS):
            if epoch < WARMUP_EPOCHS:
                lr_warmup = ((epoch + 1) / WARMUP_EPOCHS) * LEARNING_RATE
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_warmup
                if epoch + 1 == WARMUP_EPOCHS:
                    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
            lr_now = optimizer.param_groups[0]['lr']
            print(f'epoch {epoch + 1} learning rate(={round(lr_now / LEARNING_RATE * 100)}%) : {lr_now} ')
            scheduler.step()


class TesterPixelRecon(object):
    def __init__(self):
        self.model = None
        self.epochs = [0]
        self.losses = [0]
        self.model_given = None

    def process(self):
        self.build_model()
        self.eval_model()

    def build_model(self):
        self.model = shuffled_mae.__dict__['mae_vit_large_patch16_dec512d8b'](norm_pix_loss=True).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        msg = self.model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        if 'given' not in str(model_path):
            self.epochs = checkpoint['epochs']
            self.losses = checkpoint['losses']
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        print(f'Epochs: {self.epochs[-1]}')

        self.model_given = facebook_mae.__dict__['mae_vit_large_patch16_dec512d8b'](norm_pix_loss=True).to(device)
        checkpoint = torch.load(given_model_path)
        msg = self.model_given.load_state_dict(checkpoint['model'], strict=False)
        print(msg)

    def eval_model(self):
        model = self.model.eval()
        model_given = self.model_given.eval()

        this_loss = 0
        total_loss = 0
        cnt = 10
        with torch.no_grad():
            for i, data in tqdm_notebook(enumerate(test_loader, 0), total=len(test_loader)):
                samples, _ = data
                samples = samples.to(device, non_blocking=True)
                loss, pred, mask, pred_jigsaw, target_jigsaw = model(samples, mask_ratio=.75)

                this_loss = loss
                total_loss += this_loss

                print(f'(Eval Model) Loss of this images: {this_loss:.4f}')
                correct, total = acc_jigsaw(pred_jigsaw, target_jigsaw)
                print(f'acc of jigsaw: {correct / total * 100:.2f}%')
                inout_images_plot(samples=samples, mask=mask, pred=pred, model=model)

                loss, pred, mask = model_given(samples, mask_ratio=.75)
                this_loss = loss
                print(f'(Given Model) Loss of this images: {this_loss:.4f}')
                inout_images_plot(samples=samples, mask=mask, pred=pred, model=model)

                if i == cnt-1:
                    break

        loss_test = total_loss / cnt
        print(f'Avg loss of {cnt} test images: {loss_test:.4f}')

        this_loss = 0
        total_loss = 0
        cnt = 10
        with torch.no_grad():
            for i, data in tqdm_notebook(enumerate(val_loader, 0), total=len(val_loader)):
                samples, _ = data
                samples = samples.to(device, non_blocking=True)
                loss, pred, mask, pred_jigsaw, target_jigsaw = model(samples, mask_ratio=.75)

                this_loss = loss
                total_loss += this_loss

                print(f'(Eval Model) Loss of this images: {this_loss:.4f}')
                correct, total = acc_jigsaw(pred_jigsaw, target_jigsaw)
                print(f'acc of jigsaw: {correct / total * 100:.2f}%')
                inout_images_plot(samples=samples, mask=mask, pred=pred, model=model)

                loss, pred, mask = model_given(samples, mask_ratio=.75)
                this_loss = loss
                print(f'(Given Model) Loss of this images: {this_loss:.4f}')
                inout_images_plot(samples=samples, mask=mask, pred=pred, model=model)

                if i == cnt-1:
                    break

            loss_val = total_loss / cnt
        print(f'Avg loss of {cnt} val images: {loss_val:.4f}')

        print(f'Avg loss of test: {loss_test:.4f}, Avg loss of val: {loss_val:.4f}')

    def lr_checker(self):
        self.build_model()
        model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        for epoch in range(NUM_EPOCHS):
            if epoch < WARMUP_EPOCHS:
                lr_warmup = ((epoch + 1) / WARMUP_EPOCHS) * LEARNING_RATE
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_warmup
                if epoch + 1 == WARMUP_EPOCHS:
                    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
            lr_now = optimizer.param_groups[0]['lr']
            print(f'epoch {epoch + 1} learning rate(={round(lr_now / LEARNING_RATE * 100)}%) : {lr_now} ')
            scheduler.step()


if __name__ == '__main__':
    t = TesterFacebook()
    [t.process() for i in range(1)]