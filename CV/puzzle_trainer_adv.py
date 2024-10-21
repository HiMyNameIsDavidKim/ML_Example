import PIL
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from tqdm import tqdm

from CV.puzzle_fcvit_gen import FCGen, inverse_loss
from CV.puzzle_fcvit_dis import FCDis
from CV.util.tester import visualDoubleLoss, visualLoss

device = 'cpu'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''Pre-training'''
LEARNING_RATE_GEN = 3e-05
LR_RATIO = 1000
LEARNING_RATE_DIS = LEARNING_RATE_GEN / LR_RATIO
BATCH_SIZE = 2  # 64
NUM_EPOCHS = 100
NUM_WORKERS = 2
TASK_NAME = 'puzzle_imagenet'
MODEL_NAME = 'adv'
pre_model_path = f'./save/{TASK_NAME}_{MODEL_NAME}_ep{NUM_EPOCHS}_lr{LEARNING_RATE_GEN}_r{LR_RATIO}_b{BATCH_SIZE}.pt'
pre_load_model_path = './save/path.pt'
pre_reload_model_path = './save/path.pt'

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
        self.model_gen = None
        self.model_dis = None
        self.optimizer_gen = None
        self.optimizer_dis = None
        self.scheduler_gen = None
        self.scheduler_dis = None
        self.epochs = []
        self.losses_gen = []
        self.losses_dis = []
        self.accuracies = []

    def process(self, load=False, reload=False):
        self.build_model(load)
        self.pretrain_model(reload)
        self.save_model()

    def build_model(self, load):
        self.model_gen = FCGen().to(device)
        self.model_dis = FCDis().to(device)
        print(f'Parameter of gen: {sum(p.numel() for p in self.model_gen.parameters() if p.requires_grad)}')
        print(f'Parameter of dis: {sum(p.numel() for p in self.model_dis.parameters() if p.requires_grad)}')
        if load:
            checkpoint = torch.load(pre_load_model_path)
            self.epochs = checkpoint['epochs']
            if checkpoint.get('model'):
                self.model_dis.load_state_dict(checkpoint['model'])
                print(f'Parameter of dis: {sum(p.numel() for p in self.model_dis.parameters() if p.requires_grad)}')
            elif checkpoint.get('model_gen') and checkpoint.get('model_dis'):
                self.model_gen.load_state_dict(checkpoint['model_gen'])
                self.model_dis.load_state_dict(checkpoint['model_dis'])
                self.losses_gen = checkpoint['losses_gen']
                self.losses_dis = checkpoint['losses_dis']
                print(f'Parameter of gen: {sum(p.numel() for p in self.model_gen.parameters() if p.requires_grad)}')
                print(f'Parameter of dis: {sum(p.numel() for p in self.model_dis.parameters() if p.requires_grad)}')
            print(f'Epoch: {self.epochs[-1]}')
            print(f'****** Reset epochs and losses ******')
            self.epochs = []
            self.losses_gen = []
            self.losses_dis = []

    def pretrain_model(self, reload):
        model_gen = self.model_gen.train()
        model_dis = self.model_dis.train()
        criterion_gen = inverse_loss
        criterion_dis = nn.SmoothL1Loss()
        optimizer_gen = optim.AdamW(model_gen.parameters(), lr=LEARNING_RATE_GEN, weight_decay=0.05)
        optimizer_dis = optim.AdamW(model_dis.parameters(), lr=LEARNING_RATE_DIS, weight_decay=0.05)
        scheduler_gen = CosineAnnealingLR(optimizer_gen, T_max=NUM_EPOCHS)
        scheduler_dis = CosineAnnealingLR(optimizer_dis, T_max=NUM_EPOCHS)
        range_epochs = range(NUM_EPOCHS)
        if reload:
            checkpoint = torch.load(pre_reload_model_path)
            model_gen.load_state_dict(checkpoint['model_gen'])
            model_dis.load_state_dict(checkpoint['model_dis'])
            model_gen.train()
            model_dis.train()
            optimizer_gen.load_state_dict(checkpoint['optimizer_gen'])
            optimizer_dis.load_state_dict(checkpoint['optimizer_dis'])
            try:
                scheduler_gen = CosineAnnealingLR(optimizer_gen, T_max=NUM_EPOCHS)
                scheduler_dis = CosineAnnealingLR(optimizer_dis, T_max=NUM_EPOCHS)
            except:
                temp_optim_gen = optim.Adam(model_gen.parameters(), lr=LEARNING_RATE_GEN)
                temp_optim_dis = optim.Adam(model_dis.parameters(), lr=LEARNING_RATE_DIS)
                temp_scheduler_gen = CosineAnnealingLR(temp_optim_gen, T_max=NUM_EPOCHS)
                temp_scheduler_dis = CosineAnnealingLR(temp_optim_dis, T_max=NUM_EPOCHS)
                [temp_scheduler_gen.step() for _ in range(checkpoint['epochs'][-1])]
                [temp_scheduler_dis.step() for _ in range(checkpoint['epochs'][-1])]
                scheduler_gen.load_state_dict(temp_scheduler_gen.state_dict())
                scheduler_dis.load_state_dict(temp_scheduler_dis.state_dict())
            self.epochs = checkpoint['epochs']
            self.losses_gen = checkpoint['losses_gen']
            self.losses_dis = checkpoint['losses_dis']
            self.accuracies = checkpoint['accuracies']
            range_epochs = range(self.epochs[-1], NUM_EPOCHS)

        for epoch in range_epochs:
            print(f"epoch {epoch + 1} learning rate : {optimizer_gen.param_groups[0]['lr']}")
            running_loss_gen = 0.
            running_loss_dis = 0.
            for batch_idx, (inputs, _) in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                inputs = inputs.to(device)

                optimizer_gen.zero_grad()
                optimizer_dis.zero_grad()

                perm = model_gen(inputs)
                outputs, labels = model_dis(inputs, perm)

                loss_gen = criterion_gen(outputs, labels)
                loss_dis = criterion_dis(outputs, labels)
                loss_gen.backward()
                loss_dis.backward()
                optimizer_gen.step()
                optimizer_dis.step()
                running_loss_gen += loss_gen.item()
                running_loss_dis += loss_dis.item()

                inter = 100
                if batch_idx % inter == inter - 1:
                    print(f'[Epoch {epoch + 1}] [Batch {batch_idx + 1}] Gen Loss: {running_loss_gen / inter:.4f}')
                    print(f'[Epoch {epoch + 1}] [Batch {batch_idx + 1}] Dis Loss: {running_loss_dis / inter:.4f}')
                    self.epochs.append(epoch + 1)
                    self.losses_gen.append(running_loss_gen / inter)
                    self.losses_dis.append(running_loss_dis / inter)
                    running_loss_gen = 0.
                    running_loss_dis = 0.
            scheduler_gen.step()
            scheduler_dis.step()
            self.model_gen = model_gen
            self.model_dis = model_dis
            self.optimizer_gen = optimizer_gen
            self.optimizer_dis = optimizer_dis
            self.scheduler_gen = scheduler_gen
            self.scheduler_dis = scheduler_dis
            self.save_model()
            visualDoubleLoss(self.losses_gen, self.losses_dis)
            self.val_model(epoch)
        print('****** Finished Fine-tuning ******')
        self.model_gen = model_gen
        self.model_dis = model_dis

    def val_model(self, epoch=-1):
        model_dis = self.model_dis

        model_dis.eval()

        total = 0
        correct = 0
        correct_puzzle = 0
        with torch.no_grad():
            for batch_idx, (inputs, _) in tqdm(enumerate(val_loader, 0), total=len(val_loader)):
                inputs = inputs.to(device)

                outputs, labels = model_dis(inputs)

                pred = outputs
                total += labels.size(0)
                pred_ = model_dis.mapping(pred)
                labels_ = model_dis.mapping(labels)
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
            'model_gen': self.model_gen.state_dict(),
            'model_dis': self.model_dis.state_dict(),
            'optimizer_gen': self.optimizer_gen.state_dict(),
            'optimizer_dis': self.optimizer_dis.state_dict(),
            'scheduler_gen': self.scheduler_gen.state_dict(),
            'scheduler_dis': self.scheduler_dis.state_dict(),
            'losses_gen': self.losses_gen,
            'losses_dis': self.losses_dis,
            'accuracies': self.accuracies,
        }
        torch.save(checkpoint, pre_model_path)
        # if self.epochs[-1] % 50 == 0:
        #     torch.save(checkpoint, pre_model_path[:-3]+f'_{self.epochs[-1]}l{NUM_EPOCHS}.pt')
        print(f"****** Model checkpoint saved at epochs {self.epochs[-1]} ******")


if __name__ == '__main__':
    trainer = PreTrainer()
    trainer.process()
