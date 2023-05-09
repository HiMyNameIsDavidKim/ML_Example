import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from CV.util import imagenet_ind2str

device = 'mps'
BATCH_SIZE = 1
NUM_WORKERS = 2

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_set = datasets.ImageFolder('./data/ImageNet/val', transform=transform_test)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


def show_origin(n):
    for i, data in enumerate(test_loader):
        if i == n:
            inputs, labels = data
            inputs_np, labels_np = inputs.numpy(), labels.numpy()

            inputs_np = np.transpose(inputs_np, (0, 2, 3, 1))[0]
            plt.imshow(inputs_np)
            plt.title(imagenet_ind2str(int(labels_np)))
            plt.show()
            break

def show_conf(n):
    pass



patchTrans_menus = ["Exit",  # 0
                    "Show Original Image",  # 1
                    ]

patchTrans_lambda = {
    "1": lambda t: show_origin(int(input('Please input image number : '))),
    "2": lambda t: print(" ** No Function ** "),
    "3": lambda t: print(" ** No Function ** "),
    "4": lambda t: print(" ** No Function ** "),
    "5": lambda t: print(" ** No Function ** "),
    "6": lambda t: print(" ** No Function ** "),
    "7": lambda t: print(" ** No Function ** "),
    "8": lambda t: print(" ** No Function ** "),
    "9": lambda t: print(" ** No Function ** "),
}

if __name__ == '__main__':
    t = 1
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(patchTrans_menus)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("### Exit ###")
            break
        else:
            try:
                patchTrans_lambda[menu](t)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")
