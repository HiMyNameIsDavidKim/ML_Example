import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from CV.vit_paper import ViT


device = 'mps'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './save/paper_ViT_Cifar10.pt'
BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

IMAGE_SIZE = 32
PATCH_SIZE = 4
IN_CHANNELS = 3
NUM_CLASSES = 10
EMBED_DIM = 512
DEPTH = 12
NUM_HEADS = 8

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data/', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


class PaperViTCifar10Model(object):
    def __init__(self):
        self.epochs = []
        self.model = None
        self.optimizer = None
        self.losses = []
        self.cls_token = None

    def process(self):
        self.build_model()
        self.pretrain_model()
        self.save_model()
        self.eval_model()

    def build_model(self):
        self.model = ViT(image_size=IMAGE_SIZE,
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
            for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] loss: {running_loss / 100:.3f}')
            if epoch % 1 == 0:
                self.epochs.append(epoch + 1)
                self.model = model
                self.optimizer = optimizer
                self.losses.append(running_loss)
                self.save_model()
        print('****** Finished Pre-Training ******')

        self.model = model
        self.cls_token = model.cls_token

    def save_model(self):
        checkpoint = {
            'epochs': self.epochs,
            'model': self.model,
            'optimizer': self.optimizer.state_dict(),
            'losses': self.losses,
            'cls_token': self.cls_token.detach().numpy(),
        }
        torch.save(checkpoint, model_path)

    def eval_model(self):
        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the {len(testset)} test images: {100 * correct / total:.2f} %')


class Tester(object):
    def __init__(self):
        self.epoch = None
        self.model = None
        self.optimizer = None
        self.cls_token = None

    def process(self):
        self.build_model()
        self.eval_model()

    def build_model(self):
        self.model = ViT(image_size=IMAGE_SIZE,
                         patch_size=PATCH_SIZE,
                         in_channels=IN_CHANNELS,
                         num_classes=NUM_CLASSES,
                         embed_dim=EMBED_DIM,
                         depth=DEPTH,
                         num_heads=NUM_HEADS,
                         ).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)

        checkpoint = torch.load(model_path)
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.cls_token = torch.from_numpy(checkpoint['cls_token']).to(device)
        self.model.cls_token = self.cls_token

        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        print(f'epoch: {self.epoch}')

    def eval_model(self):
        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy {len(testset)} test images: {100 * correct / total:.2f} %')


if __name__ == '__main__':
    PaperViTCifar10Model().process()
