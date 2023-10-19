from CV.low_freq_mura_util import single, dim2, dim3, save_img, half_half, \
    gray, apply_noise


if __name__ == '__main__':
    # 그레이 full 패턴
    # for i in [30]:
    #     level = i
    #     gray_ptn = gray(level)
    #     save_img(gray_ptn, f'./save/gray{level}_full.png')

    # 그레이 1/4 패턴
    # for i in [65]:
    #     level = i
    #     gray_ptn = gray(level)
    #     gray_ptn_dim2 = dim2(gray_ptn)
    #     save_img(gray_ptn_dim2, f'./save/gray{level}_dim2.png')\

    # 그레이 full 패턴 + 노이즈
    for i in [30]:
        level = i
        gray_ptn = dim2(gray(level))
        intensity = 10
        gray_ptn = apply_noise(gray_ptn, intensity)
        save_img(gray_ptn, f'./save/gray{level}_full_noise{intensity}.png')

    # 그레이 반반 패턴 (1 is full, 2 is 1/4)
    # level_1 = 30
    # level_2 = 65
    # single_ptn = gray(level_1)
    # single_ptn_dim = dim2(gray(level_2, greenish=True))
    # half_half_ptn = half_half(single_ptn, single_ptn_dim)
    # save_img(half_half_ptn, f'./save/gray{level_1}_full_and_gray{level_2}_dim2.png')

    str_color = 'red'

    # 싱글 full 패턴
    # for i in [30]:
    #     level = i
    #     single_ptn = single(level, str_color)
    #     save_img(single_ptn, f'./save/{str_color}{level}_full.png')

    # 싱글 1/4 패턴
    # for i in [65]:
    #     level = i
    #     single_ptn = single(level, str_color)
    #     single_ptn_dim = dim3(single_ptn)
    #     save_img(single_ptn_dim, f'./save/{str_color}{level}_dim3.png')

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
    # single_ptn_dim = dim3(single(level_2, str_color))
    # half_half_ptn = half_half(single_ptn, single_ptn_dim)
    # save_img(half_half_ptn, f'./save/{str_color}{level_1}_full_and_{str_color}{level_2}_dim3.png')