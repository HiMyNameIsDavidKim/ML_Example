from PIL import Image
import numpy as np

width, height = 1080, 1920

def gray(level):
    image = np.full((height, width), level, dtype=np.uint8)
    return image

def save_img(image, path):
    image = Image.fromarray(image, 'L')
    image.save(path)

def quarter(image):
    image[1::2, ::] = 0
    image[::, 1::2] = 0
    return image


if __name__ == '__main__':
    gray46 = gray(46)
    save_img(gray46, './save/gray46_full.png')

    gray186 = gray(186)
    gray186_q = quarter(gray186)
    save_img(gray186_q, './save/gray186_quarter.png')


