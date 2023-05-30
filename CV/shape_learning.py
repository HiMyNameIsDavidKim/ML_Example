import numpy as np
import timm
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import seaborn as sns
import cv2
from PIL import Image

from CV.util import imagenet_ind2str

device = 'mps'
BATCH_SIZE = 1
NUM_WORKERS = 2
IMAGE_PATH = r'./data/ImageNet/shape/n01582220/ILSVRC2012_val_00000963.JPEG'

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform_origin = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])
origin_set = datasets.ImageFolder('./data/ImageNet/val', transform=transform_origin)
test_set = datasets.ImageFolder('./data/ImageNet/val', transform=transform_test)
origin_loader = DataLoader(origin_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


def ExecuteLambda(*params):
    cmd = params[0]
    target = params[1]
    if cmd == 'IMAGE_READ':
        return (lambda x: cv2.imread(x, cv2.IMREAD_COLOR))(target)
    elif cmd == 'GRAY_SCALE':
        return (lambda x: x[:, :, 0] * 0.114 + x[:, :, 1] * 0.587 + x[:, :, 2] * 0.229)(target)
    elif cmd == 'ARRAY_TO_IMAGE':
        return (lambda x: (Image.fromarray(x)))(target)
    elif cmd == '':
        pass

def Img2Edge(*params):
    img = ExecuteLambda('IMAGE_READ', params[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f'img type : {type(img)}')
    edges = cv2.Canny(np.array(img), 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(r'./save/shape/edge.png', edges)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge'), plt.xticks([]), plt.yticks([])
    plt.show()

shape_menus = ["Exit",  # 0
               "Image to Shape",  # 1
               ]

shape_lambda = {
    "1": lambda t: Img2Edge(IMAGE_PATH),
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
    t = None
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(shape_menus)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("### Exit ###")
            break
        else:
            try:
                shape_lambda[menu](t)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")
