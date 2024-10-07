import PIL
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timm.data import create_transform, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import math

from tqdm import tqdm

from CV.puzzle_vit_cls import JCViT_v2
from CV.util.tester import visualDoubleLoss


device = 'cpu'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''Pre-training'''
LEARNING_RATE = 3e-05
BATCH_SIZE = 64
NUM_EPOCHS = 20
NUM_WORKERS = 2
TASK_NAME = 'puzzle_ImageNet'
MODEL_NAME = 'jcvit_v2'
pre_load_model_path = './save/xxx.pt'
pre_model_path = f'./save/{TASK_NAME}_{MODEL_NAME}_ep{NUM_EPOCHS}_lr{LEARNING_RATE}_b{BATCH_SIZE}.pt'
pre_reload_model_path = './save/xxx.pt'


transform = transforms.Compose([
    transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.Pad(padding=(0, 0, 1, 1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
# train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# val_dataset = Subset(train_dataset, list(range(int(0.2*len(train_dataset)))))
# val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

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
        self.losses_c = []
        self.losses_cls = []
        self.losses_t = []
        self.accuracies = []

    def process(self, load=False, reload=False):
        self.build_model(load)
        self.pretrain_model(reload)
        self.save_model()

    def build_model(self, load):
        self.model = JCViT_v2().to(device)
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        if load:
            checkpoint = torch.load(pre_load_model_path)
            self.epochs = checkpoint['epochs']
            self.model.load_state_dict(checkpoint['model'])
            self.losses_c = checkpoint['losses_coord']
            self.losses_cls = checkpoint['losses_cls']
            self.losses_t = checkpoint['losses_total']
            print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
            print(f'Epoch: {self.epochs[-1]}')
            print(f'****** Reset epochs and losses ******')
            self.epochs = []
            self.losses_c = []
            self.losses_cls = []
            self.losses_t = []

    def pretrain_model(self, reload):
        model = self.model.train()
        criterion = nn.SmoothL1Loss()
        criterion_cls = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
            self.losses_c = checkpoint['losses_coord']
            self.losses_t = checkpoint['losses_total']
            self.accuracies = checkpoint['accuracies']
            range_epochs = range(self.epochs[-1], NUM_EPOCHS)

        for epoch in range_epochs:
            print(f"epoch {epoch + 1} learning rate : {optimizer.param_groups[0]['lr']}")
            running_loss_c = 0.
            running_loss_cls = 0.
            running_loss_t = 0.
            correct = 0
            total = 0
            for batch_idx, (inputs, labels_cls) in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                inputs = inputs.to(device)
                labels_cls = labels_cls.to(device)

                optimizer.zero_grad()

                outputs, labels, outputs_cls = model(inputs)

                loss_coord = criterion(outputs, labels)
                loss_cls = criterion_cls(outputs_cls, labels_cls)
                loss = loss_cls + loss_coord
                loss.backward()
                optimizer.step()

                running_loss_c += loss_coord.item()
                running_loss_cls += loss_cls.item()
                running_loss_t += loss.item()

                _, predicted = torch.max(outputs_cls.data, 1)
                total += labels_cls.size(0)
                correct += (predicted == labels_cls).sum().item()

                inter = 100
                if batch_idx % inter == inter - 1:
                    print(f'[Epoch {epoch + 1}] [Batch {batch_idx + 1}] Loss: {running_loss_c / inter:.4f}')
                    print(f'[Epoch {epoch + 1}] [Batch {batch_idx + 1}] Loss: {running_loss_cls / inter:.4f}')
                    print(f'[Epoch {epoch + 1}] [Batch {batch_idx + 1}] Total Loss: {running_loss_t / inter:.4f}')
                    self.epochs.append(epoch + 1)
                    self.losses_c.append(running_loss_c / inter)
                    self.losses_cls.append(running_loss_cls / inter)
                    self.losses_t.append(running_loss_t / inter)
                    running_loss_c = 0.
                    running_loss_cls = 0.
                    running_loss_t = 0.
                # if batch_idx % 7000 == 6999:
                #     self.val_model(epoch)
            scheduler.step()
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.save_model()
            visualDoubleLoss(self.losses_c, self.losses_cls)
            self.val_model(epoch)
        print('****** Finished Fine-tuning ******')
        self.model = model

    def val_model(self, epoch=-1):
        model = self.model

        model.eval()

        total = 0
        correct = 0
        total_cls = 0
        correct_cls = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels_cls) in tqdm(enumerate(val_loader, 0), total=len(val_loader)):
                inputs = inputs.to(device)
                labels_cls = labels_cls.to(device)

                outputs, labels, outputs_cls = model(inputs)

                pred = outputs
                total += labels.size(0)
                pred_ = model.mapping(pred)
                labels_ = model.mapping(labels)
                correct += (pred_ == labels_).all(dim=2).sum().item()

                _, predicted_cls = torch.max(outputs_cls.data, 1)
                total_cls += labels_cls.size(0)
                correct_cls += (predicted_cls == labels_cls).sum().item()

        acc = 100 * correct / (total * labels.size(1))
        print(f'[Epoch {epoch + 1}] Jigsaw accuracy on the test set: {acc:.2f}%')
        print(f'[Epoch {epoch + 1}] Classification accuracy on the test set: {100 * correct_cls / total_cls:.2f}%')

    def save_model(self):
        checkpoint = {
            'epochs': self.epochs,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'losses_coord': self.losses_c,
            'losses_cls': self.losses_cls,
            'losses_total': self.losses_t,
            'accuracies': self.accuracies,
        }
        torch.save(checkpoint, pre_model_path)
        # if self.epochs[-1] % 50 == 0:
        #     torch.save(checkpoint, pre_model_path[:-3]+f'_{self.epochs[-1]}l{NUM_EPOCHS}.pt')
        print(f"****** Model checkpoint saved at epochs {self.epochs[-1]} ******")


if __name__ == '__main__':
    trainer = PreTrainer()
    trainer.process()
