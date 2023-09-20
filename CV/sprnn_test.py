import cv2
from PIL import Image
import numpy as np
import random

width, height = 1179, 2556  # 30 and 59
# width, height = 512, 260
# width, height = 1080, 2340  # 30 and 65


def gray(level):
    color = (level, level, level)
    image = np.full((height, width, 3), color, dtype=np.uint8)
    return image

def green(level):
    color = (0, level, 0)
    image = np.full((height, width, 3), color, dtype=np.uint8)
    return image


def save_img(image, path):
    cv2.imwrite(path, image)
    # dpi = 460
    # image = image.resize((int(width * dpi / 25.4), int(height * dpi / 25.4)))
    # image.save(path, dpi=(dpi, dpi))

def dim2(image):
    image[1::2, ::] = 0
    image[::, 1::2] = 0
    return image

def dim3(image):
    image[1::3, ::] = 0
    image[2::3, ::] = 0
    image[::, 1::3] = 0
    image[::, 2::3] = 0
    return image

def apply_noise(image, intensity, green=False):
    noise = np.random.uniform(-intensity, intensity, size=image.shape)
    if green:
        noise[:,:,:1] = 0
        noise[:, :, -1:] = 0
    image = image - noise
    return image

def half_half(image_1, image_2):
    image_1 = image_1[:int(height/2),:,:]
    image_2 = image_2[int(height/2):,:,:]
    new_image = np.concatenate((image_1, image_2), axis=0)
    return new_image


if __name__ == '__main__':
    # for i in [30]:
    #     level = i
    #     gray_ptn = gray(level)
    #     save_img(gray_ptn, f'./save/gray{level}_full.png')

    # level = 186
    # gray_ptn = gray(level)
    # gray_ptn_dim2 = dim2(gray_ptn)
    # save_img(gray_ptn_dim2, f'./save/gray{level}_dim2.png')\

    # for i in [30]:
    #     level = i
    #     gray_ptn = gray(level)
    #     intensity = 6
    #     gray_ptn = apply_noise(gray_ptn, intensity)
    #     save_img(gray_ptn, f'./save/gray{level}_full_noise{intensity}.png')

    for i in [30]:
        level = i
        green_ptn = green(level)
        save_img(green_ptn, f'./save/green{level}_full.png')

    for i in [59]:
        level = i
        green_ptn = green(level)
        green_ptn_dim = dim2(green_ptn)
        save_img(green_ptn_dim, f'./save/green{level}_dim2.png')

    # for i in [65]:
    #     level = i
    #     green_ptn = green(level)
    #     green_ptn_dim = dim2(green_ptn)
    #     intensity = 4
    #     green_ptn_noise = apply_noise(green_ptn_dim, intensity, green=True)
    #     save_img(green_ptn_noise, f'./save/green{level}_dim2_noise{intensity}.png')

    # # 1 is noise, 2 is good

    level_1 = 30
    level_2 = 60
    green_ptn = green(level_1)
    green_ptn_dim = dim2(green(level_2))
    half_half_ptn = half_half(green_ptn, green_ptn_dim)
    save_img(half_half_ptn, f'./save/green{level_1}_full_and_green{level_2}_dim2.png')

    # level_1 = 65
    # level_2 = 65
    # green_ptn_noise_dim = dim2(apply_noise(green(level_1), 4, green=True))
    # green_ptn_dim = dim2(green(level_2))
    # half_half_ptn = half_half(green_ptn_noise_dim, green_ptn_dim)
    # save_img(half_half_ptn, f'./save/green{level_1}_noise4_dim2_and_green{level_2}_dim2.png')