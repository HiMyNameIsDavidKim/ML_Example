import math

import matplotlib.pyplot as plt
import numpy as np

from CV.low_freq_mura_util import single, dim2, dim3, save_img, half_half, \
    dithering_gif, dim_left_top, dim_right_top, dim_right_bottom, dim_left_bottom


if __name__ == '__main__':
    frames = []
    cnt_frame = 1
    level = 65

    # test
    # levels = list(range(90, 210, 4))
    # images = [single(level+1, 'blue') for level in levels]
    # [frames.extend([image] * cnt_frame) for image in images]

    dim_images = [dim_left_top(single(i, 'green')) for i in np.arange(65.0, -0.1, -6.5)]


    # values = np.arange(0.0, 20.1, 2.0)
    # sin = [math.sin(2 * math.pi * i / (len(values) - 1)) for i in range(len(values))]
    # weight = 0.6
    # corr_values = [i + (j * weight) for i, j in zip(values, sin)]
    full_images = [single(i, 'green') for i in range(0, 31, 3)]

    added_images = [i+j for i, j in zip(dim_images, full_images)]

    for cnt_frame in [1]:
        # dithering
        # images = [
        #     dim_left_top(single(level, 'green')),
        #     dim_right_top(single(level, 'green')),
        #     dim_right_bottom(single(level, 'green')),
        #     dim_left_bottom(single(level, 'green')),
        #           ]
        images = added_images
        [frames.extend([image] * cnt_frame) for image in images]
        [frames.extend([image] * cnt_frame) for image in images[::-1]]

        dithering_gif(frames, f'./save/gradation_div10.gif')
    # #     # dithering_gif(frames, f'./save/level{level}_dithering_60Hz_cnt{cnt_frame}.gif')
