import PIL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from timm.data import create_transform, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet50
import math
from tqdm import tqdm

from CV.puzzle_res50 import PuzzleCNNCoord
from CV.puzzle_vit import PuzzleViT
from CV.util.tester import visualDoubleLoss

import facebook_vit
from mae_util import interpolate_pos_embed
from timm.models.layers import trunc_normal_

device = 'cpu'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''Pre-training'''
LEARNING_RATE = 3e-05
BATCH_SIZE = 64
NUM_EPOCHS = 20
NUM_WORKERS = 2
TASK_NAME = 'puzzle_ImageNet'
MODEL_NAME = 'cnn50'
pre_model_path = f'./save/{TASK_NAME}_{MODEL_NAME}_ep{NUM_EPOCHS}_lr{LEARNING_RATE}_b{BATCH_SIZE}.pt'
pre_load_model_path = './save/xxx.pt'

'''Fine-tuning'''
AUGMENTATION = True
LEARNING_RATE = 3e-05
BATCH_SIZE = 64
NUM_EPOCHS = 100
WARMUP_EPOCHS = 5
NUM_WORKERS = 2
TASK_NAME = 'classification_ImageNet'
fine_load_model_path = './save/xxx.pt'  # duplicate file
fine_model_path = fine_load_model_path[:-3] + f'___{TASK_NAME}_ep{NUM_EPOCHS}_lr{LEARNING_RATE}_b{BATCH_SIZE}.pt'

'''Pre-training'''
# transform = transforms.Compose([
#     transforms.Pad(padding=3),
#     transforms.CenterCrop(30),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# transform = transforms.Compose([
#     transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
#     transforms.CenterCrop(224),
#     transforms.Pad(padding=(0, 0, 1, 1)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

'''Fine-tuning'''
transform = transforms.Compose([
    transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if AUGMENTATION:
    transform = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=None,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )

mixup_fn = Mixup(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    cutmix_minmax=None,
    prob=1.0,
    switch_prob=0.5,
    mode='batch',
    label_smoothing=0.1,
    num_classes=1000
)

# train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
# train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# val_dataset = Subset(train_dataset, list(range(int(0.2*len(train_dataset)))))
# val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

train_dataset = datasets.ImageFolder('../datasets/ImageNet/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dataset = Subset(train_dataset, list(range(int(0.01 * len(train_dataset)))))
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dataset = datasets.ImageFolder('../datasets/ImageNet/val', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


class PreTrainer(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.epochs = []
        self.losses_c = []
        self.losses_t = []

    def process(self, load=False):
        self.build_model(load)
        self.pretrain_model()
        self.save_model()

    def build_model(self, load):
        self.model = PuzzleCNNCoord().to(device)
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        if load:
            checkpoint = torch.load(pre_load_model_path)
            self.epochs = checkpoint['epochs']
            self.model.load_state_dict(checkpoint['model'])
            self.losses_c = checkpoint['losses_coord']
            self.losses_t = checkpoint['losses_total']
            print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
            print(f'Epoch: {self.epochs[-1]}')
            print(f'****** Reset epochs and losses ******')
            self.epochs = []
            self.losses_c = []
            self.losses_t = []

    def pretrain_model(self):
        model = self.model
        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        model.train()
        for epoch in range(NUM_EPOCHS):
            running_loss_c = 0.
            running_loss_t = 0.
            for batch_idx, (inputs, _) in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                inputs = inputs.to(device)

                optimizer.zero_grad()

                outputs, labels, loss_var = model(inputs)
                loss_coord = criterion(outputs, labels)
                loss = loss_coord + loss_var / 1e05
                loss.backward()
                optimizer.step()
                running_loss_c += loss_coord.item()
                running_loss_t += loss.item()

                inter = 100
                if batch_idx % inter == inter - 1:
                    print(f'[Epoch {epoch + 1}] [Batch {batch_idx + 1}] Loss: {running_loss_c / inter:.4f}')
                    print(f'[Epoch {epoch + 1}] [Batch {batch_idx + 1}] Total Loss: {running_loss_t / inter:.4f}')
                    self.epochs.append(epoch + 1)
                    self.model = model
                    self.optimizer = optimizer
                    self.losses_c.append(running_loss_c / inter)
                    self.losses_t.append(running_loss_t / inter)
                    running_loss_c = 0.
                    running_loss_t = 0.
                if batch_idx % 7000 == 6999:
                    self.val_model(epoch)
            scheduler.step()
            self.save_model()
            visualDoubleLoss(self.losses_c, self.losses_t)
            self.val_model(epoch)
        print('****** Finished Fine-tuning ******')
        self.model = model

    def val_model(self, epoch=-1):
        model = self.model

        model.eval()

        total = 0
        diff = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, _) in tqdm(enumerate(val_loader, 0), total=len(val_loader)):
                inputs = inputs.to(device)

                outputs, labels, _ = model(inputs)

                pred = outputs
                total += labels.size(0)
                diff += (torch.dist(pred, labels)).sum().item()
                pred_ = model.mapping(pred)
                labels_ = model.mapping(labels)
                correct += (pred_ == labels_).all(dim=2).sum().item()

        print(f'[Epoch {epoch + 1}] Avg diff on the test set: {diff / total:.2f}')
        print(f'[Epoch {epoch + 1}] Accuracy on the test set: {100 * correct / (total * labels.size(1)):.2f}%')
        torch.set_printoptions(precision=2)
        total = labels.size(1)
        correct = (pred_[0] == labels_[0]).all(dim=1).sum().item()
        print(f'[Sample result]')
        print(torch.cat((pred_[0], labels_[0]), dim=1))
        print(f'Accuracy: {100 * correct / total:.2f}%')

    def save_model(self):
        checkpoint = {
            'epochs': self.epochs,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses_coord': self.losses_c,
            'losses_total': self.losses_t,
        }
        torch.save(checkpoint, pre_model_path)
        print(f"****** Model checkpoint saved at epochs {self.epochs[-1]} ******")


class FineTuner(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epochs = [0]
        self.losses = [0]
        self.accuracies = [0]

    def process(self, load=False):
        self.build_model(load)
        self.finetune_model()
        self.save_model()

    def build_model(self, load):
        self.model = facebook_vit.__dict__['vit_base_patch16'](
            num_classes=1000,
            drop_path_rate=0.1,
            global_pool=True,
        )
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        self.optimizer = optim.SGD(self.model.parameters(), lr=0)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=NUM_EPOCHS)

        if load:
            checkpoint = torch.load(fine_load_model_path, map_location=device)
            checkpoint_model = checkpoint['model']
            for key in list(checkpoint_model.keys()):
                if key.startswith('vit_features.'):
                    new_key = key.replace('vit_features.', '')
                    checkpoint_model[new_key] = checkpoint_model.pop(key)
            for key in list(checkpoint_model.keys()):
                if key.startswith('norm.'):
                    new_key = key.replace('norm.', 'fc_norm.')
                    checkpoint_model[new_key] = checkpoint_model.pop(key)

            state_dict = self.model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            interpolate_pos_embed(self.model, checkpoint_model)
            msg = self.model.load_state_dict(checkpoint_model, strict=False)
            print(msg)
            trunc_normal_(self.model.head.weight, std=2e-5)
            self.model.to(device)

            if 'given' not in str(fine_load_model_path):
                self.epochs = checkpoint['epochs']
            print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
            print(f'Epoch: {self.epochs[-1]}')
            print(f'****** Reset epochs and losses ******')
            self.epochs = []
            self.losses = []
            self.accuracies = []

    def finetune_model(self):
        model = self.model.train()
        criterion = nn.CrossEntropyLoss()
        if AUGMENTATION:
            criterion = SoftTargetCrossEntropy()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        for epoch in range(NUM_EPOCHS):
            if epoch < WARMUP_EPOCHS:
                lr_warmup = ((epoch + 1) / WARMUP_EPOCHS) * LEARNING_RATE
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_warmup
                if epoch + 1 == WARMUP_EPOCHS:
                    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
            print(f"epoch {epoch + 1} learning rate : {optimizer.param_groups[0]['lr']}")
            running_loss = 0.0
            saving_loss = 0.0
            correct = 0
            total = 0
            for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                if AUGMENTATION:
                    inputs, labels = mixup_fn(inputs, labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                saving_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                if not AUGMENTATION:
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                inter = 100
                if i % inter == inter - 1:
                    if AUGMENTATION:
                        print(f'[Epoch {epoch}, Batch {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    else:
                        print(f'[Epoch {epoch}, Batch {i + 1:5d}] loss: {running_loss / 100:.3f}, acc: {correct / total * 100:.2f} %')
                        self.accuracies.append(correct / total * 100)
                    self.epochs.append(epoch + 1)
                    self.losses.append(saving_loss / 1000)
                    running_loss = 0.0
                    saving_loss = 0.0
                    correct = 0
                    total = 0
                if i % 7000 == 6999:
                    self.val_model(epoch)
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.save_model()
            self.val_model(epoch)
            scheduler.step()
        print('****** Finished Fine-tuning ******')

    def val_model(self, epoch=-1):
        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader)):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'[Epoch {epoch + 1}] Accuracy of {len(val_dataset)} test images: {100 * correct / total:.2f} %')

    def save_model(self):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epochs': self.epochs,
            'losses': self.losses,
            'accuracies': self.accuracies,
        }
        torch.save(checkpoint, fine_model_path)
        #         torch.save(checkpoint, dynamic_model_path+str(self.epochs[-1])+f'_lr{LEARNING_RATE}.pt')
        print(f"****** Model checkpoint saved at epochs {self.epochs[-1]} ******")


if __name__ == '__main__':
    trainer = PreTrainer()
    trainer.process()
