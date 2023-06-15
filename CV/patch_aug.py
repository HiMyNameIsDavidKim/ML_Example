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


class CutMix:
    def __init__(self, beta=1.0):
        self.beta = beta

    def __call__(self, images, labels):
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        lam = torch.FloatTensor([self.beta])
        lam = lam.to(images.device)
        bbx1, bby1, bbx2, bby2 = self._generate_bbox(images.size(2), images.size(3), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
        labels = lam * labels + (1 - lam) * labels[indices]
        return images, labels

    def _generate_bbox(self, image_width, image_height, lam):
        cut_ratio = torch.sqrt(1. - lam)
        cut_w = (image_width * cut_ratio).type(torch.long)
        cut_h = (image_height * cut_ratio).type(torch.long)
        cx = torch.randint(0, image_width, (1,))
        cy = torch.randint(0, image_height, (1,))
        bbx1 = torch.clamp(cx - cut_w // 2, 0, image_width)
        bby1 = torch.clamp(cy - cut_h // 2, 0, image_height)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, image_width)
        bby2 = torch.clamp(cy + cut_h // 2, 0, image_height)
        return bbx1, bby1, bbx2, bby2


class PositivePatchShuffle(object):
    def __init__(self, p=0.5, p_size=32):
        self.p = p
        self.p_size = p_size

    def __call__(self, img):
        if random.random() > self.p:
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
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass



# if __name__ == '__main__':
#     device = 'mps'
#     BATCH_SIZE = 1
#     NUM_WORKERS = 2
#
#     transform_test = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         PositivePatchShuffle(p=1),
#         transforms.ToTensor(),
#     ])
#     test_set = datasets.ImageFolder('./data/ImageNet/val', transform=transform_test)
#     test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
#
#     for idx, (images, labels) in enumerate(test_loader):
#         if idx == 0:
#             images = images.numpy()
#             images = np.transpose(images, (0, 2, 3, 1))[0]
#             plt.imshow(images)
#             plt.show()
#             break
