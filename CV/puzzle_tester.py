import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50
import math
from tqdm import tqdm

from CV.puzzle_res50 import PuzzleCNNCoord
from CV.util.tester import visualDoubleLoss


device = 'cpu'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 3e-05
BATCH_SIZE = 64
NUM_EPOCHS = 20
test_model_path = './save/xxx.pt'


transform = transforms.Compose([
    transforms.Pad(padding=3),
    transforms.CenterCrop(30),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class Tester(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.epochs = [0]
        self.losses_c = [0]
        self.losses_t = [0]

    def process(self, load=False):
        self.build_model(load)
        self.eval_model()

    def build_model(self, load):
        self.model = PuzzleCNNCoord().to(device)
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        if load:
            checkpoint = torch.load(test_model_path)
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

    def eval_model(self, epoch=-1):
        model = self.model

        model.eval()

        total = 0
        diff = 0
        correct = 0
        with torch.no_grad():
            for inputs, _ in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
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
        print(torch.cat((pred_[0], labels_[0]), dim=1))
        print(f'Accuracy: {100 * correct / total:.2f}')


if __name__ == '__main__':
    tester = Tester()
    tester.process()
