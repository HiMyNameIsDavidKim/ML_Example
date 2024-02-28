import math

import torch
from skimage.io import imread
from torchvision import transforms as T
import matplotlib.pyplot as plt


# train = 2040 x 2040
# val = 2040 x 2040
# test = 3456 x 5184


def pad_img(img, device, img_size=5184):
    img = T.ToTensor()(img.copy()).to(device=device)
    c, h, w = img.shape
    pad_h = img_size - h
    pad_w = img_size - w
    top_pad = pad_h // 2
    bottom_pad = pad_h - top_pad
    left_pad = pad_w // 2
    right_pad = pad_w - left_pad

    transforms_img = T.Compose([
        T.Pad((left_pad, top_pad, right_pad, bottom_pad), fill=0)
    ])
    img = transforms_img(img)

    img = img.unsqueeze(0)

    return img

def crop_img(img, origin_img, device, img_size=5184):
    origin_img = T.ToTensor()(origin_img.copy()).to(device=device)
    c, h, w = origin_img.shape
    pad_h = img_size - h
    pad_w = img_size - w
    top_pad = pad_h // 2
    bottom_pad = pad_h - top_pad
    left_pad = pad_w // 2
    right_pad = pad_w - left_pad

    img = img.squeeze(0)

    img = img[:, top_pad:img_size - bottom_pad, left_pad:img_size - right_pad]

    return img

def split_img(img, device, img_size=192):
    img = T.ToTensor()(img.copy()).to(device=device)
    c, h, w = img.shape
    num_H = math.ceil(h/img_size)
    num_W = math.ceil(w/img_size)
    H = num_H*img_size
    W = num_W*img_size
    pad_h = H - h
    pad_w = W - w
    top_pad = pad_h//2
    bottom_pad = pad_h-top_pad
    left_pad = pad_w//2
    right_pad = pad_w-left_pad

    transforms_img = T.Compose([
        T.Pad((left_pad, top_pad, right_pad, bottom_pad), fill=0)
    ])
    img = transforms_img(img)

    pieces = []
    for i in range(num_H):
        for j in range(num_W):
            start_row = i * img_size
            start_col = j * img_size
            piece = img[:, start_row:start_row + img_size,
                    start_col:start_col + img_size]
            pieces.append(piece)
    pieces = torch.stack(pieces)

    return pieces

def concat_pieces(pieces, origin_img, device, img_size=192):
    origin_img = T.ToTensor()(origin_img.copy()).to(device=device)
    c, h, w = origin_img.shape
    num_H = math.ceil(h/img_size)
    num_W = math.ceil(w/img_size)
    H = num_H * img_size
    W = num_W * img_size
    pad_h = H - h
    pad_w = W - w
    top_pad = pad_h // 2
    bottom_pad = pad_h - top_pad
    left_pad = pad_w // 2
    right_pad = pad_w - left_pad
    img = torch.zeros([3, H, W])
    index = 0
    for i in range(num_H):
        for j in range(num_W):
            start_row = i * img_size
            start_col = j * img_size
            img[:, start_row:start_row + img_size, start_col:start_col + img_size] = pieces[index]
            index += 1
    img = img[:, top_pad:H-bottom_pad, left_pad:W-right_pad]
    return img


if __name__ == '__main__':
    img = imread('../data/rabbit.jpg')
    print(img.shape)
    img2 = pad_img(img, device='cpu')
    print(img2.shape)
    img3 = crop_img(img2, img, device='cpu')
    print(img3.shape)

    img2 = img2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img3 = img3.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img3)
    plt.show()

    # img = imread('../data/rabbit.jpg')
    # print(img.shape)
    # pieces = split_img(img, device='cpu')
    # img2 = concat_pieces(pieces, img, device='cpu')
    #
    # img2 = img2.permute(1, 2, 0).cpu().numpy()
    # plt.imshow(img)
    # plt.show()