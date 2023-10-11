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

    for cnt_frame in [1, 2, 3, 4]:
        # dithering
        single_ptn = single(level, 'blue')
        images = [dim_left_top(single_ptn),
                  dim_right_top(single_ptn),
                  dim_right_bottom(single_ptn),
                  dim_left_bottom(single_ptn),
                  ]
        [frames.extend([image] * cnt_frame) for image in images]

        dithering_gif(frames, f'./save/level{level}_dithering_60Hz_cnt{cnt_frame}.gif')
