import torch
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets


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


class PositivePatchShuffle(object):
    def __init__(self, p=0.5, p_size=32):
        self.p = p
        self.p_size = p_size

    def __call__(self, img):
        if torch.rand() > self.p:
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


# 클래스 단위에서 하는게 안됨. (loss를 건드려야함.)
# negative loss regularization 참고하기.
# criterion 같은거 다 불러와서 처리.

class NegativePatchShuffle(object):
    def __init__(self, p=0.5, p_size=32):
        self.p = p
        self.p_size = p_size


if __name__ == '__main__':
    device = 'mps'
    BATCH_SIZE = 1
    NUM_WORKERS = 2

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        PositivePatchShuffle(),
        transforms.ToTensor(),
    ])
    test_set = datasets.ImageFolder('./data/ImageNet/val', transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = self.model
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    shuffler = NegativePatchShuffle()

    for idx, data in enumerate(test_loader):
        if idx == 0:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
