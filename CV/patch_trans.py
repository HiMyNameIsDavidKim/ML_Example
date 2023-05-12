import numpy as np
import timm
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

# 데이터셋 커스텀 transform, 데이터셋 합치기
# class YourDataset2(Dataset):
#     def __init__(self):
#         pass
#
#     def __getitem__(self, idx):
#         img = self.data[idx]
#         transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Lambda(shuffler),
#             transforms.ToTensor(),
#         ])
#         img = transform(img)
#         label = self.labels[idx]
#         return img, label
#
# concat_dataset = ConcatDataset([dataset1, dataset2])
# concat_dataloader = DataLoader(concat_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

def shuffler(img):
    d = 7
    sub_imgs = []
    for i in range(d):
        for j in range(d):
            sub_img = img[i * 224 // d:(i + 1) * 224 // d, j * 224 // d:(j + 1) * 224 // d]
            sub_imgs.append(sub_img)
    np.random.shuffle(sub_imgs)
    new_img = np.vstack([np.hstack([sub_imgs[i] for i in range(d*j, d*(j+1))]) for j in range(d)])
    return new_img


def rotator(img):
    d = 7
    sub_imgs = []
    for i in range(d):
        for j in range(d):
            sub_img = img[i * 224 // d:(i + 1) * 224 // d, j * 224 // d:(j + 1) * 224 // d]
            sub_imgs.append(sub_img)
    sub_imgs = [np.rot90(sub_img) for sub_img in sub_imgs]
    new_img = np.vstack([np.hstack([sub_imgs[i] for i in range(d * j, d * (j + 1))]) for j in range(d)])
    return new_img


def show_img(n, shuffle=False, rotate=False):
    for i, data in enumerate(origin_loader):
        if i == n:
            inputs, labels = data
            inputs_np, labels_np = inputs.numpy(), labels.numpy()
            inputs_np = np.transpose(inputs_np, (0, 2, 3, 1))[0]
            if shuffle:
                inputs_np = shuffler(inputs_np)
            if rotate:
                inputs_np = rotator(inputs_np)
            plt.imshow(inputs_np)
            plt.title(imagenet_ind2str(int(labels_np)))
            plt.show()
            break


def cal_conf(n, shuffle=False, rotate=False):
    model = timm.models.vit_base_patch16_224(pretrained=True)
    print(f'Parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(f'Classes: {model.num_classes}')
    print(f'****** Model Creating Completed. ******')
    model.to(device).eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            if idx == n:
                images = images.numpy()
                images = np.transpose(images, (0, 2, 3, 1))[0]
                if shuffle:
                    images = shuffler(images)
                if rotate:
                    images = rotator(images)
                images = torch.from_numpy(images.transpose((2, 0, 1)).reshape(1, 3, 224, 224)).float()
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                _, pred = torch.max(outputs, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                conf = probs[int(labels)].to('cpu')

                print(f'Label : {imagenet_ind2str(int(labels))}')
                print(f'Predict : {imagenet_ind2str(int(pred))}')
                print(f'Confidence of label : {float(conf):.3f}')
                break


patchTrans_menus = ["Exit",  # 0
                    "Show Image(Original)",  # 1
                    "Calculate Confidence of Image(Original)",  # 2
                    "Show Image(Shuffle)",  # 3
                    "Calculate Confidence of Image(Shuffle)",  # 4
                    "Show Image(Rotate)",  # 5
                    "Calculate Confidence of Image(Rotate)",  # 6
                    ]

patchTrans_lambda = {
    "1": lambda t: show_img(int(input('Please input image number : '))),
    "2": lambda t: cal_conf(int(input('Please input image number : '))),
    "3": lambda t: show_img(int(input('Please input image number : ')), shuffle=True),
    "4": lambda t: cal_conf(int(input('Please input image number : ')), shuffle=True),
    "5": lambda t: show_img(int(input('Please input image number : ')), rotate=True),
    "6": lambda t: cal_conf(int(input('Please input image number : ')), rotate=True),
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
