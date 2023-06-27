import timm
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets


class PositivePatchShuffle(object):
    def __init__(self, p=0.5, p_size=32):
        self.p = p
        self.p_size = p_size

    def __call__(self, img):
        if np.random.random() > self.p:
            return img

        img = np.array(img)
        i_size = img.shape[0]
        d = int(i_size/self.p_size)
        sub_imgs = []
        for i in range(d):
            for j in range(d):
                sub_img = img[i * 224 // d:(i + 1) * 224 // d, j * 224 // d:(j + 1) * 224 // d]
                sub_imgs.append(sub_img)
        np.random.shuffle(sub_imgs)
        new_img = np.vstack([np.hstack([sub_imgs[i] for i in range(d * j, d * (j + 1))]) for j in range(d)])
        return F.to_pil_image(new_img)


class NegativePatchShuffle(object):
    def __init__(self, p=0.5, p_size=32):
        self.p = p
        self.p_size = p_size
        self.turn_on = False
        self.coefficient = 1

    def roll_the_dice(self):
        if np.random.random() > self.p:
            self.turn_on = False
        else:
            self.turn_on = True

    def shuffle(self, imgs):
        if self.turn_on:
            imgs = imgs.numpy()
            imgs = np.transpose(imgs, (0, 2, 3, 1))
            batch_size, height, width, channels = imgs.shape
            d = int(height / self.p_size)
            new_imgs = []
            for img in imgs:
                sub_imgs = []
                for i in range(d):
                    for j in range(d):
                        sub_img = img[i * 224 // d:(i + 1) * 224 // d, j * 224 // d:(j + 1) * 224 // d]
                        sub_imgs.append(sub_img)
                np.random.shuffle(sub_imgs)
                new_img = np.vstack([np.hstack([sub_imgs[i] for i in range(d * j, d * (j + 1))]) for j in range(d)])
                new_imgs.append(new_img)
            new_imgs = np.stack(new_imgs)
            new_imgs = torch.from_numpy(new_imgs.transpose((0, 3, 1, 2))).float()
            # sample = F.to_pil_image(new_imgs[0])
            # plt.imshow(sample)
            # plt.show()
            return new_imgs
        else:
            return imgs

    def cal_loss(self, outputs, labels, criterion, device):
        loss_ce = criterion(outputs, labels)
        if self.turn_on:
            max_ind = torch.tensor([i.argmax() for i in outputs]).to(device)
            loss_neg = criterion(outputs, max_ind)/1000
            print(loss_ce, loss_neg)
            return loss_ce + (self.coefficient * loss_neg)
        else:
            return loss_ce


class Cutout(object):
    def __init__(self, p=0.5, cutout_ratio=0.2):
        self.p = p
        self.cutout_ratio = cutout_ratio

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img = F.to_tensor(img)
        h, w = img.shape[-2:]
        cutout_size = int(min(w, h) * self.cutout_ratio)
        left = random.randint(0, w - cutout_size)
        upper = random.randint(0, h - cutout_size)
        img[:, upper:upper + cutout_size, left:left + cutout_size] = 0.0
        return F.to_pil_image(img)


class MixUp(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        batch_size = image.size(0)
        indices = torch.randperm(batch_size)
        image2, label2 = image[indices], label[indices]
        lam = torch.distributions.beta.Beta(self.alpha, self.alpha).sample((batch_size,)).to(image.device)
        lam = torch.max(lam, 1 - lam)
        mixed_image = lam.view(batch_size, 1, 1, 1) * image + (1 - lam).view(batch_size, 1, 1, 1) * image2
        mixed_label = lam * label + (1 - lam) * label2
        return {'image': mixed_image, 'label': mixed_label}


if __name__ == '__main__':
    device = 'mps'
    BATCH_SIZE = 4
    NUM_WORKERS = 2

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # PositivePatchShuffle(),
        transforms.ToTensor(),
    ])
    test_set = datasets.ImageFolder('./data/ImageNet/val', transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = timm.models.vit_base_patch16_224(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    aug = NegativePatchShuffle(p=1)

    for idx, data in enumerate(test_loader):
        if idx == 0:
            inputs, labels = data
            aug.roll_the_dice()
            inputs = aug.shuffle(inputs)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = aug.cal_loss(outputs, labels, criterion, device)
            loss.backward()
            optimizer.step()
            break