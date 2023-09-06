from PIL import Image
import numpy as np

width, height = 10, 10

def gray(level):
    image = np.full((height, width), level, dtype=np.uint8)
    return image

def save_img(image, path):
    image = Image.fromarray(image, 'L')
    image.save(path)
    # dpi = 460
    # image = image.resize((int(width * dpi / 25.4), int(height * dpi / 25.4)))
    # image.save(path, dpi=(dpi, dpi))

def dim2(image):
    image[1::2, ::] = 0
    image[::, 1::2] = 0
    print(image)
    return image

def dim3(image):
    image[1::3, ::] = 0
    image[2::3, ::] = 0
    image[::, 1::3] = 0
    image[::, 2::3] = 0
    return image


if __name__ == '__main__':
    level = 46
    gray46 = gray(level)
    # save_img(gray46, f'./save/gray{level}_full.png')

    level = 186
    gray186 = gray(level)
    gray186_dim2 = dim2(gray186)
    # save_img(gray186_dim2, f'./save/gray{level}_dim2.png')

    level = 186
    gray186 = gray(level)
    gray186_dim3 = dim3(gray186)
    # save_img(gray186_dim3, f'./save/gray{level}_dim3.png')
