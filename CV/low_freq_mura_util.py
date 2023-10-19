import math

import cv2
from PIL import Image
import numpy as np
import random

# width, height = 1179, 2556  # 30 and 59
# width, height = 512, 260
width, height = 1080, 2340  # 30 and 65
# width, height = 2,2


def gray(level, greenish=False):
    color = (level, level, level)
    if greenish:
        color = (level, level*1.1, level)
    image = np.full((height, width, 3), color, dtype=np.uint8)
    return image

def single(level, str_color):
    if str_color =='green':
        color = (0, level, 0)
    elif str_color =='blue':
        color = (0, 0, level)
    elif str_color =='red':
        color = (level, 0, 0)
    image = np.full((height, width, 3), color, dtype=np.uint8)
    return image

def save_img(image, path):
    image = image[..., ::-1]
    cv2.imwrite(path, image)
    # dpi = 460
    # image = image.resize((int(width * dpi / 25.4), int(height * dpi / 25.4)))
    # image.save(path, dpi=(dpi, dpi))

def dim2(image):
    image[1::2, ::] = 0
    image[::, 1::2] = 0
    return image

def dimRoot2(image):
    distance = math.sqrt(2)
    image[1::2, ::] = 0
    image[::, 1::2] = 0
    return image


def dim_left_top(image_origin):
    image = np.copy(image_origin)
    image[1::2, ::] = 0
    image[::, 1::2] = 0
    return image

def dim_right_top(image_origin):
    image = np.copy(image_origin)
    image[::2, ::] = 0
    image[::, 1::2] = 0
    return image

def dim_right_bottom(image_origin):
    image = np.copy(image_origin)
    image[::2, ::] = 0
    image[::, ::2] = 0
    return image

def dim_left_bottom(image_origin):
    image = np.copy(image_origin)
    image[1::2, ::] = 0
    image[::, ::2] = 0
    return image


def dim3(image):
    image[1::3, ::] = 0
    image[2::3, ::] = 0
    image[::, 1::3] = 0
    image[::, 2::3] = 0
    return image

def apply_noise(image, intensity):
    level = [i for i in image[0][0].flatten() if i != 0][0]
    intensity = level * (intensity/100)
    noise = np.random.uniform(-intensity, intensity, size=image.shape)
    image = image - noise
    return image

def half_half(image_1, image_2):
    image_1 = image_1[:int(height/2),:,:]
    image_2 = image_2[int(height/2):,:,:]
    new_image = np.concatenate((image_1, image_2), axis=0)
    return new_image

def dithering_gif(frames, save_path):
    frames = [Image.fromarray(frame) for frame in frames]
    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=16.67, loop=0)


