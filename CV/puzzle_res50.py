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

from CV.util.tester import visualDoubleLoss

# --------------------------------------------------------
# PuzzleCNN
# img_size=30, patch_size=10, num_puzzle=9
# input = [batch, 3, 30, 30]
# shuffle
# dim_resnet = [batch, 2048]
# dim_fc = [batch, 4096]
# output = [batch, 9, 2]
# --------------------------------------------------------

device = 'cpu'
lr = 3e-05
batch_size = 64
NUM_EPOCHS = 20


transform = transforms.Compose([
    transforms.Pad(padding=3),
    transforms.CenterCrop(30),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class PuzzleCNNCoord(nn.Module):
    def __init__(self, num_puzzle=9, size_puzzle=10, threshold=0.8):
        super(PuzzleCNNCoord, self).__init__()
        self.num_puzzle = num_puzzle
        self.size_puzzle = size_puzzle
        self.threshold = threshold
        resnet = resnet50(pretrained=True)
        resnet_output_size = resnet.fc.in_features
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        self.fc1 = nn.Linear(resnet_output_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, self.num_puzzle * 2)
        self.map_values = []
        self.map_coord = None
        self.min_dist = 0

    def random_shuffle(self, x):
        N, C, H, W = x.shape
        p = self.size_puzzle
        n = int(math.sqrt(self.num_puzzle))

        noise = torch.rand(N, self.num_puzzle, device=x.device)
        ids_shuffles = torch.argsort(noise, dim=1)
        ids_restores = torch.argsort(ids_shuffles, dim=1)

        for i, (img, ids_shuffle) in enumerate(zip(x, ids_shuffles)):
            pieces = [img[:, i:i + p, j:j + p] for i in range(0, H, p) for j in range(0, W, p)]
            shuffled_pieces = [pieces[idx] for idx in ids_shuffle]
            shuffled_img = [torch.cat(row, dim=2) for row in [shuffled_pieces[i:i+n] for i in range(0, len(shuffled_pieces), n)]]
            shuffled_img = torch.cat(shuffled_img, dim=1)
            x[i] = shuffled_img

        start, end = 0, n
        self.min_dist = (end-start)/n
        self.map_values = list(torch.arange(start, end, self.min_dist))
        self.map_coord = torch.tensor([(i, j) for i in self.map_values for j in self.map_values])

        coord_shuffles = torch.zeros([N, self.num_puzzle, 2])
        coord_restores = torch.zeros([N, self.num_puzzle, 2])
        for i, (ids_shuffle, ids_restore) in enumerate(zip(ids_shuffles, ids_restores)):
            coord_shuffles[i] = self.map_coord[ids_shuffle]
            coord_restores[i] = self.map_coord[ids_restore]

        return x, coord_restores.to(x.device)

    def forward_loss_var(self, x):
        N, n, c = x.shape
        self_distances = torch.zeros((N, n, n), device=x.device)
        for batch in range(N):
            self_distances[batch] = torch.cdist(x[batch], x[batch]) + torch.eye(self.num_puzzle, device=x.device)
        loss_var = torch.sum(torch.relu((self.threshold * self.min_dist) - self_distances))
        return loss_var

    def mapping(self, target):
        diff = torch.abs(target.unsqueeze(3) - torch.tensor(self.map_values, device=target.device))
        min_indices = torch.argmin(diff, dim=3)
        target[:] = min_indices
        return target

    def forward(self, x):
        x, target = self.random_shuffle(x)

        x = self.resnet_features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = x.view(-1, self.num_puzzle, 2)

        loss_var = self.forward_loss_var(x)

        return x, target, loss_var


if __name__ == '__main__':
    model = PuzzleCNNCoord().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    ls = []
    ls_t = []
    for epoch in range(NUM_EPOCHS):
        model.train()

        running_loss = 0.
        running_loss_t = 0.
        for batch_idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)

            outputs, labels, loss_var = model(inputs)

            optimizer.zero_grad()

            loss_coord = criterion(outputs, labels)
            print(f'loss coord : {loss_coord}')
            print(f'loss_var : {loss_var}')
            loss = loss_coord + loss_var/1e05

            loss.backward()
            optimizer.step()

            running_loss += loss_coord.item()
            running_loss_t += loss.item()

            inter = 30
            if batch_idx % inter == inter-1:
                print(f'[Epoch {epoch+1}] [Batch {batch_idx+1}] Loss: {running_loss/inter:.4f}')
                print(f'[Epoch {epoch+1}] [Batch {batch_idx+1}] Total Loss: {running_loss_t/inter:.4f}')
                ls.append(running_loss/inter)
                ls_t.append(running_loss_t/inter)
                running_loss = 0.
                running_loss_t = 0.
        scheduler.step()
        visualDoubleLoss(ls, ls_t)

        model.eval()

        total = 0
        diff = 0
        correct = 0
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)

                outputs, labels, _ = model(inputs)

                pred = outputs
                total += labels.size(0)
                diff += (torch.dist(pred, labels)).sum().item()
                pred_ = model.mapping(pred)
                labels_ = model.mapping(labels)
                correct += (pred_ == labels_).all(dim=2).sum().item()

        print(f'[Epoch {epoch+1}] Avg diff on the test set: {diff / total:.2f}')
        print(f'[Epoch {epoch+1}] Accuracy on the test set: {100 * correct / (total*labels.size(1)):.2f}%')
        torch.set_printoptions(precision=2)
        total = labels.size(1)
        correct = (pred_[0] == labels_[0]).all(dim=1).sum().item()
        print(torch.cat((pred_[0], labels_[0]), dim=1))
        print(f'Accuracy: {100 * correct / (labels.size(1)):.2f}')

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'losses': ls,
            'total_losses': ls_t,
        }
        torch.save(checkpoint, f'./save/puzzle_res50_ep{NUM_EPOCHS}_lr{lr}_b{batch_size}.pt')
        print(f"### Model checkpoint saved at epoch {epoch+1} ###")
