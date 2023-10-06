from CV.low_freq_mura_util import single, dim2, dim3, save_img, half_half, dithering_gif


if __name__ == '__main__':
    frames = []
    cnt_frame = 2

    levels = list(range(90, 120))
    images = [single(level+1, 'blue') for level in levels]
    [frames.extend([image] * cnt_frame) for image in images]

    dithering_gif(frames, f'./save/gif_test.gif')
