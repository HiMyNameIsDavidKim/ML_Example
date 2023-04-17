import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm


model_path = './save/timm_ViT_Cifar100.pt'
# device = 'mps'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 512
NUM_EPOCHS = 10000
NUM_WORKERS = 2
LEARNING_RATE = 0.001

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
trainset = datasets.CIFAR100(root='./data/', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
testset = datasets.CIFAR100(root='./data/', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


class ViTCifar100Model(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epochs = []
        self.losses = []

    def process(self):
        self.build_modeL()
        self.train_model()
        self.eval_model()

    def build_modeL(self):
        self.model = timm.models.vit_base_patch16_224(pretrained=True).to(device)
        # self.model = timm.models.vit_large_patch16_224(pretrained=True).to(device)
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')

    def train_model(self):
        model = self.model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0, last_epoch=-1)

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
                self.scheduler = scheduler
                self.losses.append(running_loss)
                self.save_model()
            scheduler.step()
            print('****** Finished Training ******')

    def save_model(self):
        checkpoint = {
            'epoch': self.epochs,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'losses': self.losses,
        }
        torch.save(checkpoint, model_path)
        print(f"****** Model checkpoint saved at epoch {self.epochs[-1]} ******")

    def eval_model(self):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
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


class Tester(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epochs = []
        self.losses = []

    def process(self):
        self.build_model()
        self.eval_model()

    def build_model(self):
        self.model = timm.models.vit_base_patch16_224(pretrained=True).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)

        checkpoint = torch.load(model_path)
        self.epochs = checkpoint['epochs']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.losses = checkpoint['losses']
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        print(f'epoch: {self.epochs[-1]}')

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


if __name__ == "__main__":
    ViTCifar100Model().process()