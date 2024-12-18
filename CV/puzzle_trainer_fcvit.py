import PIL
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from tqdm import tqdm

from CV.puzzle_fcvit_3x3 import FCViT
from CV.util.tester import visualDoubleLoss, visualLoss

device = 'cpu'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''Pre-training'''
LEARNING_RATE = 3e-05
BATCH_SIZE = 64
NUM_EPOCHS = 20
NUM_WORKERS = 2
TASK_NAME = 'puzzle_ImageNet'
MODEL_NAME = 'fcvit'
pre_load_model_path = './save/xxx.pt'
pre_model_path = f'./save/{TASK_NAME}_{MODEL_NAME}_ep{NUM_EPOCHS}_lr{LEARNING_RATE}_b{BATCH_SIZE}.pt'
pre_reload_model_path = './save/xxx.pt'

'''Pre-training'''
transform = transforms.Compose([
    transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.Pad(padding=(0, 0, 1, 1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('../datasets/ImageNet/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
val_dataset = Subset(train_dataset, list(range(int(0.01 * len(train_dataset)))))
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
test_dataset = datasets.ImageFolder('../datasets/ImageNet/val', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)


class PreTrainer(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epochs = []
        self.losses = []
        self.accuracies = []

    def process(self, load=False, reload=False):
        self.build_model(load)
        self.pretrain_model(reload)
        self.save_model()

    def build_model(self, load):
        self.model = FCViT().to(device)
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

    def pretrain_model(self, reload):
        model = self.model.train()
        criterion = nn.SmoothL1Loss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        range_epochs = range(NUM_EPOCHS)
        if reload:
            checkpoint = torch.load(pre_reload_model_path)
            model.load_state_dict(checkpoint['model'])
            model.train()
            optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
            except:
                temp_optim = optim.Adam(model.parameters(), lr=LEARNING_RATE)
                temp_scheduler = CosineAnnealingLR(temp_optim, T_max=NUM_EPOCHS)
                [temp_scheduler.step() for _ in range(checkpoint['epochs'][-1])]
                scheduler.load_state_dict(temp_scheduler.state_dict())
            self.epochs = checkpoint['epochs']
            self.losses = checkpoint['losses']
            self.accuracies = checkpoint['accuracies']
            range_epochs = range(self.epochs[-1], NUM_EPOCHS)

        for epoch in range_epochs:
            print(f"epoch {epoch + 1} learning rate : {optimizer.param_groups[0]['lr']}")
            running_loss = 0.
            for batch_idx, (inputs, _) in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                inputs = inputs.to(device)

                optimizer.zero_grad()

                outputs, labels = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                inter = 100
                if batch_idx % inter == inter - 1:
                    print(f'[Epoch {epoch + 1}] [Batch {batch_idx + 1}] Loss: {running_loss / inter:.4f}')
                    self.epochs.append(epoch + 1)
                    self.losses.append(running_loss / inter)
                    running_loss = 0.
            scheduler.step()
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
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
        correct_puzzle = 0
        with torch.no_grad():
            for batch_idx, (inputs, _) in tqdm(enumerate(val_loader, 0), total=len(val_loader)):
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
        self.accuracies.append(acc)

    def save_model(self):
        checkpoint = {
            'epochs': self.epochs,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'losses': self.losses,
            'accuracies': self.accuracies,
        }
        torch.save(checkpoint, pre_model_path)
        # if self.epochs[-1] % 50 == 0:
        #     torch.save(checkpoint, pre_model_path[:-3]+f'_{self.epochs[-1]}l{NUM_EPOCHS}.pt')
        print(f"****** Model checkpoint saved at epochs {self.epochs[-1]} ******")


if __name__ == '__main__':
    trainer = PreTrainer()
    trainer.process()
