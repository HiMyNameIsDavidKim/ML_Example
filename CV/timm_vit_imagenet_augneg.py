import timm
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.optim import AdamW, SGD
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from patch_aug import NegativePatchShuffle, NegativePatchRotate

device = 'mps'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
NUM_EPOCHS = 8
NUM_WORKERS = 2
LEARNING_RATE = 0.003
pre_model_path = './save/ViT_timm_vit_base_patch16_224_in21k.pt'
fine_model_path = f'./save/ViT_timm_vit_base_patch16_224_in21k_Negative_i2012_ep{NUM_EPOCHS}_lr{LEARNING_RATE}.pt'

NUM_CLASSES = 1000

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
# pre_train_set = torchvision.datasets.ImageFolder('./data/ImageNet-21k', transform=transform_train)
# pre_train_loader = data.DataLoader(pre_train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
train_set = torchvision.datasets.ImageFolder('./data/ImageNet/train', transform=transform_train)
train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_set = torchvision.datasets.ImageFolder('./data/ImageNet/val', transform=transform_test)
test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


class FineTunner(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.epochs = [0]
        self.losses = [0]

    def process(self, load=False):
        self.build_model(load)
        self.finetune_model(load)
        self.save_model()

    def build_model(self, load):
        self.model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True).to(device)
        self.model.num_classes = NUM_CLASSES
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        print(f'Classes: {self.model.num_classes}')
        self.optimizer = SGD(self.model.parameters(), lr=0)
        if load:
            checkpoint = torch.load(pre_model_path)
            self.epochs = checkpoint['epochs']
            self.model.load_state_dict(checkpoint['model'])
            self.losses = checkpoint['losses']
            self.optimizer = checkpoint['optimizer']
            print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
            print(f'Classes: {self.model.num_classes}')
            print(f'Epoch: {self.epochs[-1]}')
            print(f'****** Reset epochs and losses ******')
            self.epochs = []
            self.losses = []

    def finetune_model(self, load):
        model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        if load:
            optimizer = self.optimizer
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        aug = NegativePatchRotate(p=0.5)

        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0
            saving_loss = 0.0
            for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                inputs, labels = data
                aug.roll_the_dice(len(inputs))
                inputs = aug.rotate(inputs)
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = aug.cal_loss(outputs, labels, criterion, device)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                saving_loss += loss.item()
                if i % 100 == 99:
                    print(f'[Epoch {epoch}, Batch {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                if i % 1000 == 999:
                    self.epochs.append(epoch + 1)
                    self.model = model
                    self.optimizer = optimizer
                    self.losses.append(saving_loss/1000)
                    self.save_model()
                    saving_loss = 0.0
            scheduler.step()
        print('****** Finished Fine-tuning ******')
        self.model = model

    def save_model(self):
        checkpoint = {
            'epochs': self.epochs,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.losses,
        }
        torch.save(checkpoint, fine_model_path)
        print(f"****** Model checkpoint saved at epochs {self.epochs[-1]} ******")


if __name__ == '__main__':
    trainer = FineTunner()
    trainer.process(load=True)