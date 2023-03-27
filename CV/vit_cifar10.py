import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from CV.vit_paper import ViT


device = 'mps'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './save/ViT_Cifar10.pt'
BATCH_SIZE = 512
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


class ViTCifar10Model(object):
    def __init__(self):
        self.model = None

    def process(self):
        self.pretrain_model()  # Check pre-training setting.
        self.train_model()
        self.save_model()
        self.eval_model()

    def pretrain_model(self):
        model = ViT(image_size=IMAGE_SIZE,
                    patch_size=PATCH_SIZE,
                    in_channels=IN_CHANNELS,
                    num_classes=NUM_CLASSES,
                    embed_dim=EMBED_DIM,
                    depth=DEPTH,
                    num_heads=NUM_HEADS,
                    ).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0
            for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
                inputs, _ = data
                inputs = inputs.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
        print('****** Finished Pre-Training ******')

        self.model = model

    def train_model(self):
        model = self.model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
                if i % 100 == 99:
                    print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
        print('****** Finished Training ******')

        self.model = model

    def save_model(self):
        torch.save(self.model, model_path)

    def eval_model(self):
        model = torch.load(model_path).to(device)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the {len(testset)} test images: {100 * correct / total:.2f} %')


if __name__ == '__main__':
    ViTCifar10Model().process()