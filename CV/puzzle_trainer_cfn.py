import PIL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from itertools import permutations

from CV.puzzle_cfn import PuzzleCFN_30, PuzzleCFN
from CV.puzzle_image_loader import PuzzleDataset1000 as PuzzleDataset
from CV.util.tester import visualLoss


device = 'cpu'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CLASSES = 1000
LEARNING_RATE = 1e-03  # 1e-03
BATCH_SIZE = 1  # 256
NUM_EPOCHS = 20
NUM_WORKERS = 2
TASK_NAME = 'puzzle_ImageNet'
MODEL_NAME = 'cfn1000'
pre_model_path = f'./save/{TASK_NAME}_{MODEL_NAME}_ep{NUM_EPOCHS}_lr{LEARNING_RATE}_b{BATCH_SIZE}.pt'
pre_load_model_path = './save/xxx.pt'


# transform = transforms.Compose([
#     transforms.Pad(padding=3),
#     transforms.CenterCrop(30),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Pad(padding=(0, 0, 1, 1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# transform = transforms.Compose([
#     transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
#     transforms.CenterCrop(224),
#     transforms.Pad(padding=(0, 0, 1, 1)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_dataset = PuzzleDataset(dataset=train_dataset)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dataset = Subset(train_dataset, list(range(int(0.2*len(train_dataset)))))
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_dataset = PuzzleDataset(dataset=test_dataset)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# train_dataset = datasets.ImageFolder('./data/ImageNet/val', transform=transform)
# # train_dataset = datasets.ImageFolder('../datasets/ImageNet/test', transform=transform)
# train_dataset = PuzzleDataset(dataset=train_dataset)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# val_dataset = Subset(train_dataset, list(range(int(0.01*len(train_dataset)))))
# val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# test_dataset = datasets.ImageFolder('./data/ImageNet/val', transform=transform)
# # test_dataset = datasets.ImageFolder('../datasets/ImageNet/val', transform=transform)
# test_dataset = PuzzleDataset(dataset=test_dataset)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


class PreTrainer(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.epochs = []
        self.losses = []
        numbers = list(range(9))
        permutation_list = permutations(numbers)
        permutations_array = np.array(list(permutation_list))
        self.permutations = permutations_array

    def process(self, load=False):
        self.build_model(load)
        self.pretrain_model()
        self.save_model()

    def build_model(self, load):
        self.model = PuzzleCFN(classes=CLASSES).to(device)
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        if load:
            checkpoint = torch.load(pre_load_model_path)
            self.epochs = checkpoint['epochs']
            self.model.load_state_dict(checkpoint['model'])
            self.losses = checkpoint['losses']
            print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
            print(f'Epoch: {self.epochs[-1]}')
            print(f'****** Reset epochs and losses ******')
            self.epochs = []
            self.losses = []

    def pretrain_model(self):
        model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        model.train()
        for epoch in range(NUM_EPOCHS):
            running_loss = 0.
            for batch_idx, (images, labels, original) in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                inter = 50
                if batch_idx % inter == inter - 1:
                    print(f'[Epoch {epoch + 1}] [Batch {batch_idx + 1}] Loss: {running_loss / inter:.4f}')
                    self.epochs.append(epoch + 1)
                    self.model = model
                    self.optimizer = optimizer
                    self.losses.append(running_loss / inter)
                    running_loss = 0.
            scheduler.step()
            self.save_model()
            visualLoss(self.losses)
            self.val_model(epoch)
        print('****** Finished Fine-tuning ******')
        self.model = model

    def val_model(self, epoch=-1):
        model = self.model

        model.eval()

        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (images, labels, original) in tqdm(enumerate(val_loader, 0), total=len(val_loader)):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, pred = torch.max(outputs.data, 1)
                labels_ = torch.tensor([self.idx2order1000(label) for label in labels])
                pred_ = torch.tensor([self.idx2order1000(p) for p in pred])
                total += labels_.size(0) * labels_.size(1)
                correct += (pred_ == labels_).sum().item()

        print(f'[Epoch {epoch + 1}] Accuracy on the test set: {100 * correct / total:.2f}%')
        torch.set_printoptions(precision=2)
        total = labels_.size(1)
        correct = (pred_[0] == labels_[0]).sum().item()
        print(f'[Sample result]')
        print(torch.cat((pred_[0].view(9, -1), labels_[0].view(9, -1)), dim=1))
        print(f'Accuracy: {100 * correct / total:.2f}%')

    def save_model(self):
        checkpoint = {
            'epochs': self.epochs,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.losses,
        }
        torch.save(checkpoint, pre_model_path)
        print(f"****** Model checkpoint saved at epochs {self.epochs[-1]} ******")

    def idx2order(self, idx):
        numbers = list(range(9))
        permutation_list = permutations(numbers)
        permutations_array = np.array(list(permutation_list))
        return permutations_array[idx]

    def idx2order1000(self, idx):
        permutations_array = np.load(f'./data/permutations_1000.npy')
        return permutations_array[idx]


if __name__ == '__main__':
    trainer = PreTrainer()
    trainer.process()

