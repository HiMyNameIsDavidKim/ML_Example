import PIL
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import math
from tqdm import tqdm

from CV.puzzle_fcvit_3x3 import FCViT


device = 'cpu'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-05
BATCH_SIZE = 512
NUM_EPOCHS = 100
NUM_WORKERS = 2
test_model_path = './save/path.pt'


transform = transforms.Compose([
    transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.Pad(padding=(0, 0, 1, 1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('../datasets/ImageNet/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dataset = Subset(train_dataset, list(range(int(0.01*len(train_dataset)))))
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dataset = datasets.ImageFolder('../datasets/ImageNet/val', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


class Tester(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.epochs = [0]
        self.losses = [0]
        self.accuracies = [0]

    def process(self, load=True):
        self.build_model(load)
        self.eval_model()

    def build_model(self, load):
        self.model = FCViT().to(device)
        self.model.augment_tile = transforms.Compose([
            transforms.RandomCrop(64),
            transforms.Resize((75, 75)),
        ])
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        if load:
            checkpoint = torch.load(test_model_path)
            self.epochs = checkpoint['epochs']
            self.model.load_state_dict(checkpoint['model'])
            self.losses = checkpoint['losses']
            self.accuracies = checkpoint['accuracies']
            print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
            print(f'Epoch: {self.epochs[-1]}')

    def eval_model(self, epoch=-1):
        model = self.model

        model.eval()

        total = 0
        correct = 0
        correct_puzzle = 0
        with torch.no_grad():
            for batch_idx, (inputs, _) in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
                inputs = inputs.to(device)

                outputs, labels = model(inputs)

                pred = outputs
                total += labels.size(0)
                pred_ = model.mapping(pred)
                labels_ = model.mapping(labels)
                correct += (pred_ == labels_).all(dim=2).sum().item()
                correct_puzzle += (pred_ == labels_).all(dim=2).all(dim=1).sum().item()

        acc = 100 * correct / (total * labels.size(1))
        acc_puzzle = 100 * correct_puzzle / (total)
        print(f'[Epoch {epoch + 1}] Accuracy (Fragment-level) on the test set: {acc:.2f}%')
        print(f'[Epoch {epoch + 1}] Accuracy (Puzzle-level) on the test set: {acc_puzzle:.2f}%')
        torch.set_printoptions(precision=2)
        total = labels.size(1)
        correct = (pred_[0] == labels_[0]).all(dim=1).sum().item()
        print(f'[Sample result]')
        print(torch.cat((pred_[0], labels_[0]), dim=1))
        print(f'Accuracy: {100 * correct / total:.2f}%')

    def loss_checker(self):
        checkpoint = torch.load(test_model_path, map_location='cpu')
        self.epochs = checkpoint['epochs']
        self.losses = checkpoint['losses']
        print(f'Steps: {len(self.epochs)} (x100) steps')
        ls_x = []
        ls_y = []
        for x, y in enumerate(self.losses):
            ls_x.append(x)
            ls_y.append(y)
        plt.plot(ls_x, ls_y)
        plt.title('losses Plot')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        for i, acc in enumerate(self.losses):
            print(f'[Epochs {i + 1}] Average Loss: {acc:.3f}')

    def lr_checker(self):
        self.build_model(load=True)
        model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        ls_epoch = []
        ls_lr = []
        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0
            saving_loss = 0.0
            ls_epoch.append(epoch)
            ls_lr.append(optimizer.param_groups[0]["lr"])
            scheduler.step()
        plt.plot(ls_epoch, ls_lr)
        plt.title('LR Plot')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.show()


if __name__ == '__main__':
    tester = Tester()
    tester.process()
