import torch
from torchvision.transforms import functional as F
import random

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


class PositivePatchShuffle(object):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass