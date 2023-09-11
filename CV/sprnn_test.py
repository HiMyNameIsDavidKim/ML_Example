import cv2
from PIL import Image
import numpy as np

width, height = 1179, 2556

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


if __name__ == '__main__':
    # level = 46
    # gray46 = gray(level)
    # save_img(gray46, f'./save/gray{level}_full.png')

    # level = 186
    # gray186 = gray(level)
    # gray186_dim2 = dim2(gray186)
    # save_img(gray186_dim2, f'./save/gray{level}_dim2.png')

    # level = 186
    # gray186 = gray(level)
    # gray186_dim3 = dim3(gray186)
    # save_img(gray186_dim3, f'./save/gray{level}_dim3.png')

    # level = 16
    # green16 = green(level)
    # save_img(green16, f'./save/green{level}_full.png')

    level = 30
    green255 = green(level)
    green255_dim = dim2(green255)
    save_img(green255_dim, f'./save/green{level}_dim2.png')
