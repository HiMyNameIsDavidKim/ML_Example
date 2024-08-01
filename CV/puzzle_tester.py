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

from CV.puzzle_vit_preTrue import PuzzleViT
from CV.puzzle_res50_preTrue import PuzzleCNNCoord
from CV.util.tester import visualDoubleLoss


device = 'cpu'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
        self.model = PuzzleViT().to(device)
        self.model.augment_tile = transforms.Compose([])
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
            for batch_idx, (inputs, _) in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
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
        print(f'[Sample result]')
        print(torch.cat((pred_[0], labels_[0]), dim=1))
        print(f'Accuracy: {100 * correct / total:.2f}%')

    def visual_model(self):
        self.build_model(True)
        model = self.model

        model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, _) in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
                inputs = inputs.to(device)

                outputs, labels, _, shuffled_inputs = model(inputs)

                visualization(labels, model, outputs, shuffled_inputs)

def visualization(labels, model, outputs, shuffled_inputs):
    H, W = 225, 225
    p = 75
    n = int(math.sqrt(9))
    start, end = 0, n
    min_dist = (end - start) / n
    map_values = list(torch.arange(start, end, min_dist))
    map_coord = torch.tensor([(i, j) for i in map_values for j in map_values])
    ids_pred = torch.zeros([9], dtype=torch.long)
    ids_label = torch.zeros([9], dtype=torch.long)
    for i in range(9):
        coord_pred = model.mapping(outputs)[0][i].to('cpu')
        coord_label = labels[0][i].to('cpu')
        index_pred = (map_coord == coord_pred).all(dim=1).nonzero(as_tuple=True)[0].item()
        index_label = (map_coord == coord_label).all(dim=1).nonzero(as_tuple=True)[0].item()
        ids_pred[i] = index_pred
        ids_label[i] = index_label
    shuffled_input = shuffled_inputs[0]
    pieces = [shuffled_input[:, i:i + p, j:j + p] for i in range(0, H, p) for j in range(0, W, p)]
    pieces_pred = [pieces[idx] for idx in ids_pred]
    pieces_label = [pieces[idx] for idx in ids_label]
    img_pred = [torch.cat(row, dim=2) for row in [pieces_pred[i:i + n] for i in range(0, len(pieces_pred), n)]]
    img_label = [torch.cat(row, dim=2) for row in [pieces_label[i:i + n] for i in range(0, len(pieces_label), n)]]
    img_pred = torch.cat(img_pred, dim=1)
    img_label = torch.cat(img_label, dim=1)
    img_input = shuffled_input.permute(1, 2, 0).cpu().numpy()
    img_pred = img_pred.permute(1, 2, 0).cpu().numpy()
    img_label = img_label.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_input = img_input * std + mean
    img_pred = img_pred * std + mean
    img_label = img_label * std + mean
    imgs = [img_input, img_pred, img_label]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.05)
    for i, ax in enumerate(axs):
        ax.imshow(imgs[i])
        ax.axis('off')
    plt.show()


def loss_checker(self):
    checkpoint = torch.load(test_model_path, map_location='cpu')
    self.epochs = checkpoint['epochs']
    self.losses_c = checkpoint['losses_coord']
    self.losses_t = checkpoint['losses_total']
    print(f'Steps: {len(self.epochs)} steps')
    ls_x = []
    ls_y = []
    for x, y in enumerate(self.losses_t):
        ls_x.append(x)
        ls_y.append(y)
    plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(ls_x, ls_y)
    plt.xlabel('Steps', fontsize=10)
    plt.ylabel('Training Loss', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(range(0, 20001, 4000))

    plt.show()
    [print(f'Average Loss: {i:.3f}') for i in self.losses_t]

def acc_checker(self):
    checkpoint = torch.load(test_model_path, map_location='cpu')
    self.epochs = checkpoint['epochs']
    self.acc = checkpoint['accuracies']
    ls_x = []
    ls_y = []
    for x, y in enumerate(self.acc):
        ls_x.append(x)
        ls_y.append(y)
    plt.figure(figsize=(4, 3), dpi=200)
    plt.plot(ls_x, ls_y)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Validation Accuracy', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()

def lr_checker(self):
    self.build_model(load=True)
    model = self.model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    ls_epoch = []
    ls_lr = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        saving_loss = 0.0
        ls_epoch.append(epoch)
        ls_lr.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    plt.plot(ls_epoch, ls_lr)
    plt.title('LR Plot')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.show()

def lr_compare(self):
    LEARNING_RATE = 1e-05
    NUM_EPOCHS = 100
    self.build_model(load=True)
    model = self.model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    ls_epoch = []
    ls_lr = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        saving_loss = 0.0
        ls_epoch.append(epoch)
        ls_lr.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    plt.plot(ls_epoch, ls_lr, label='Curve 1')

    LEARNING_RATE = 2e-05
    NUM_EPOCHS = 50
    model = self.model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    ls_epoch = []
    ls_lr = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        saving_loss = 0.0
        ls_epoch.append(epoch)
        ls_lr.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    plt.plot(ls_epoch, ls_lr, label='Curve 2')

    plt.title('LR Plot')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    tester = Tester()
    tester.process()
