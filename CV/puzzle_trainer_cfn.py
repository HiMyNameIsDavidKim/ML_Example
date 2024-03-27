import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from CV.puzzle_cfn import PuzzleCFN


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
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

# 로더 안에 넣어야함.
# 쪼개서 넣는 로더를 소스코드에서 발췌해서 복제 해줘야함.
# puzzle trainer CFN 만들기.




