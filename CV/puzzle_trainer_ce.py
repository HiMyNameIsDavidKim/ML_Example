import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet50
import math
from tqdm import tqdm

from CV.puzzle_res50_ce import PuzzleCNNCoord
from CV.puzzle_vit_ce import PuzzleViT
from CV.util.tester import visualDoubleLoss


device = 'cpu'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 3e-05
BATCH_SIZE = 64
NUM_EPOCHS = 20
NUM_WORKERS = 2
TASK_NAME = 'puzzle_cifar10'
MODEL_NAME = 'res50_ce'
pre_model_path = f'./save/{TASK_NAME}_{MODEL_NAME}_ep{NUM_EPOCHS}_lr{LEARNING_RATE}_b{BATCH_SIZE}.pt'
pre_load_model_path = './save/xxx.pt'


transform = transforms.Compose([
    transforms.Pad(padding=3),
    transforms.CenterCrop(30),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dataset = Subset(train_dataset, list(range(int(0.2*len(train_dataset)))))
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


class PreTrainer(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.epochs = [0]
        self.losses_c = [0]
        self.losses_t = [0]

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
        criterion = nn.CrossEntropyLoss()
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
                loss = loss_coord + loss_var/1e05
                loss.backward()
                optimizer.step()
                running_loss_c += loss_coord.item()
                running_loss_t += loss.item()

                inter = 100
                inter = 1
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
                break
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


if __name__ == '__main__':
    trainer = PreTrainer()
    trainer.process()