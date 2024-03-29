import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from CV.puzzle_cfn import PuzzleCFN
from CV.puzzle_image_loader import PuzzleDataLoader

device = 'cpu'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 3e-05
BATCH_SIZE = 64
NUM_EPOCHS = 20
NUM_WORKERS = 2
TASK_NAME = 'puzzle_cifar10'
MODEL_NAME = 'cfn'
pre_model_path = f'./save/{TASK_NAME}_{MODEL_NAME}_ep{NUM_EPOCHS}_lr{LEARNING_RATE}_b{BATCH_SIZE}.pt'
pre_load_model_path = './save/xxx.pt'

transform = transforms.Compose([
    transforms.Pad(padding=3),
    transforms.CenterCrop(30),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = PuzzleDataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

if __name__ == '__main__':
    for batch_idx, (images, labels, original) in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
        cat_image = torch.zeros(3, 30, 30)
        for i in range(3):
            for j in range(3):
                tile = images[i * 3 + j]
                cat_image[:, i * 10:i * 10 + 10, j * 10:j * 10 + 10] = tile

        plt.imshow(cat_image.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.show()

        original = torch.stack(original, 0)
        cat_image = torch.zeros(3, 30, 30)
        for i in range(3):
            for j in range(3):
                tile = original[i * 3 + j]
                cat_image[:, i * 10:i * 10 + 10, j * 10:j * 10 + 10] = tile

        plt.imshow(cat_image.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.show()

        break


# 정합성....? 이상한거 같은데... 섞는 알고리즘 내꺼로 변경하고 PIL 왔다갔다 빼버리기.




