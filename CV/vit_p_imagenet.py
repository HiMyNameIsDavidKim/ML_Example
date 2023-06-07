import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from keras.optimizers import Adam
from torch import nn
from tqdm import tqdm

from CV.vit_GAP import ViTPooling

device = 'mps'
pre_model_path = './save/ViT_i21k_ep300_lr0.001_augVanilla.pt'
fine_model_path = './save/ViT_i21k_ep300_lr0.001_augVanilla_i2012_step20k_lr0.03.pt'
BATCH_SIZE = 64
NUM_EPOCHS = 300
NUM_WORKERS = 2
LEARNING_RATE = 0.001

IMAGE_SIZE = 224
PATCH_SIZE = 16
IN_CHANNELS = 3
NUM_CLASSES = 1000
EMBED_DIM = 512
DEPTH = 12
NUM_HEADS = 8

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
pre_train_set = torchvision.datasets.ImageFolder('./data/ImageNet-21k', transform=transform_train)
pre_train_loader = data.DataLoader(pre_train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
train_set = torchvision.datasets.ImageFolder('./data/ImageNet/train', transform=transform_train)
train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_set = torchvision.datasets.ImageFolder('./data/ImageNet/val', transform=transform_test)
test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


class PreTrainer(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.epochs = []
        self.losses = []

    def process(self):
        self.build_model()
        self.pretrain_model()
        self.save_model()

    def build_model(self):
        self.model = ViTPooling(image_size=IMAGE_SIZE,
                                patch_size=PATCH_SIZE,
                                in_channels=IN_CHANNELS,
                                num_classes=NUM_CLASSES,
                                embed_dim=EMBED_DIM,
                                depth=DEPTH,
                                num_heads=NUM_HEADS,
                                ).to(device)

    def pretrain_model(self):
        model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0
            for i, data in tqdm(enumerate(pre_train_loader, 0), total=len(pre_train_loader)):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'[Epoch {epoch}, Batch {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
            self.epochs.append(epoch + 1)
            self.model = model
            self.optimizer = optimizer
            self.losses.append(running_loss)
            self.save_model()
        print('****** Finished Pre-training ******')

        self.model = model

    def save_model(self):
        checkpoint = {
            'epochs': self.epochs,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.losses,
        }
        torch.save(checkpoint, pre_model_path)
        print(f"****** Model checkpoint saved at epochs {self.epochs[-1]} ******")


class FineTunner(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.epochs = []
        self.losses = []

    def process(self):
        self.build_model()
        self.finetune_model()
        self.save_model()

    def build_model(self):
        self.model = ViTPooling(image_size=IMAGE_SIZE,
                                patch_size=PATCH_SIZE,
                                in_channels=IN_CHANNELS,
                                num_classes=NUM_CLASSES,
                                embed_dim=EMBED_DIM,
                                depth=DEPTH,
                                num_heads=NUM_HEADS,
                                ).to(device)
        checkpoint = torch.load(pre_model_path)
        self.epochs = checkpoint['epochs']
        self.model.load_state_dict(checkpoint['model'])
        self.losses = checkpoint['losses']
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        print(f'Classes: {NUM_CLASSES}')
        print(f'Epoch: {self.epochs[-1]}')

    def finetune_model(self):
        model = self.model
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0
            for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'[Epoch {epoch}, Batch {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
            self.epochs.append(epoch + 1)
            self.model = model
            self.optimizer = optimizer
            self.losses.append(running_loss)
            self.save_model()
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
