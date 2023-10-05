import cv2
from PIL import Image
import numpy as np
import random

# width, height = 1179, 2556  # 30 and 59
# width, height = 512, 260
width, height = 1080, 2340  # 30 and 65


def gray(level, greenish=False):
    color = (level, level, level)
    if greenish:
        color = (level, level+7, level)
    image = np.full((height, width, 3), color, dtype=np.uint8)
    return image

def single(level, str_color):
    if str_color =='green':
        color = (0, level, 0)
    elif str_color =='red':
        color = (0, 0, level)
    elif str_color =='blue':
        color = (level, 0, 0)
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

def apply_noise(image, intensity, single=False):
    noise = np.random.uniform(-intensity, intensity, size=image.shape)
    if single:
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
    # 그레이 full 패턴
    for i in [30]:
        level = i
        gray_ptn = gray(level)
        save_img(gray_ptn, f'./save/gray{level}_full.png')

    # 그레이 1/4 패턴
    for i in [65]:
        level = i
        gray_ptn = gray(level)
        gray_ptn_dim2 = dim2(gray_ptn)
        save_img(gray_ptn_dim2, f'./save/gray{level}_dim2.png')\

    # 그레이 full 패턴 + 노이즈
    # for i in [30]:
    #     level = i
    #     gray_ptn = gray(level)
    #     intensity = 6
    #     gray_ptn = apply_noise(gray_ptn, intensity)
    #     save_img(gray_ptn, f'./save/gray{level}_full_noise{intensity}.png')

    # 그레이 반반 패턴 (1 is full, 2 is 1/4)
    level_1 = 30
    level_2 = 65
    single_ptn = gray(level_1)
    single_ptn_dim = dim2(gray(level_2, greenish=True))
    half_half_ptn = half_half(single_ptn, single_ptn_dim)
    save_img(half_half_ptn, f'./save/gray{level_1}_full_and_gray{level_2}_dim2.png')

    str_color = 'blue'

    # 싱글 full 패턴
    # for i in [30]:
    #     level = i
    #     single_ptn = single(level, str_color)
    #     save_img(single_ptn, f'./save/{str_color}{level}_full.png')

    # 싱글 1/4 패턴
    # for i in [65]:
    #     level = i
    #     single_ptn = single(level, str_color)
    #     single_ptn_dim = dim2(single_ptn)
    #     save_img(single_ptn_dim, f'./save/{str_color}{level}_dim2.png')

    # 싱글 full 패턴 + 노이즈
    # for i in [65]:
    #     level = i
    #     single_ptn = single(level)
    #     single_ptn_dim = dim2(single_ptn)
    #     intensity = 4
    #     single_ptn_noise = apply_noise(single_ptn_dim, intensity, single=True)
    #     save_img(single_ptn_noise, f'./save/single{level}_dim2_noise{intensity}.png')

    # 싱글 반반 패턴 (1 is full, 2 is 1/4)
    # level_1 = 30
    # level_2 = 65
    # single_ptn = single(level_1, str_color)
    # single_ptn_dim = dim2(single(level_2, str_color))
    # half_half_ptn = half_half(single_ptn, single_ptn_dim)
    # save_img(half_half_ptn, f'./save/{str_color}{level_1}_full_and_{str_color}{level_2}_dim2.png')