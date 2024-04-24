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
from torchvision.models import resnet50
import math
from tqdm import tqdm

from CV.puzzle_cfn import PuzzleCFN_30, PuzzleCFN
from CV.puzzle_image_loader import PuzzleDataset1000 as PuzzleDataset
from CV.util.tester import visualLoss

device = 'cpu'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CLASSES = 1000
LEARNING_RATE = 3e-05
BATCH_SIZE = 64
NUM_EPOCHS = 20
NUM_WORKERS = 32
test_model_path = './save/xxx.pt'


transform = transforms.Compose([
    transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.Pad(padding=(0, 0, 1, 1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('../datasets/ImageNet/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dataset = Subset(train_dataset, list(range(int(0.01*len(train_dataset)))))
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dataset = datasets.ImageFolder('../datasets/ImageNet/val', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


class Tester(object):
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.epochs = [0]
        self.losses_c = [0]
        self.losses_t = [0]

    def process(self, load=True):
        self.build_model(load)
        self.eval_model()

    def build_model(self, load):
        self.model = PuzzleCFN(classes=CLASSES).to(device)
        print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        if load:
            checkpoint = torch.load(test_model_path)
            self.epochs = checkpoint['epochs']
            self.model.load_state_dict(checkpoint['model'])
            self.losses = checkpoint['losses']
            print(f'Parameter: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
            print(f'Epoch: {self.epochs[-1]}')
            print(f'****** Reset epochs and losses ******')
            self.epochs = []
            self.losses = []

    def eval_model(self, epoch=-1):
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

    def idx2order1000(self, idx):
        permutations_array = np.load(f'./data/permutations_1000.npy')
        return permutations_array[idx]


if __name__ == '__main__':
    tester = Tester()
    tester.process()
