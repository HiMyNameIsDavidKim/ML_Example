import cv2
from PIL import Image
import numpy as np

# width, height = 1179, 2556
width, height = 512, 260

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
    for i in [4, 8, 16, 30, 46, 64]:
        level = i
        gray_ptn = gray(level)
        save_img(gray_ptn, f'./save/gray{level}_full.png')

    level = 186
    gray_ptn = gray(level)
    gray_ptn_dim2 = dim2(gray_ptn)
    save_img(gray_ptn_dim2, f'./save/gray{level}_dim2.png')

    level = 186
    gray_ptn = gray(level)
    gray_ptn_dim3 = dim3(gray_ptn)
    save_img(gray_ptn_dim3, f'./save/gray{level}_dim3.png')

    for i in [4, 8, 16, 30, 46, 64]:
        level = i
        green_ptn = green(level)
        save_img(green_ptn, f'./save/green{level}_full.png')

    level = 30
    green_ptn = green(level)
    green_ptn_dim = dim2(green_ptn)
    save_img(green_ptn_dim, f'./save/green{level}_dim2.png')
