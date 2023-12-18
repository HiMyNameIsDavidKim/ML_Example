import numpy as np
from PIL.Image import Image


def read_image(file_path):
    image = Image.open(file_path)
    image_array = np.array(image)
    return image_array

def copy_shape(copy_target, value, save_path):
    shape = copy_target.shape
    image_array = np.full(shape, value, dtype=np.uint8)
    new_image = Image.fromarray(image_array)
    new_image.save(save_path)

