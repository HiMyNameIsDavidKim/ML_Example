import time

import torch

from puzzle_fcvit_4x4 import FCViT
from puzzle_cfn import PuzzleCFN
from puzzle_jpdvt import JPDVT, get_2d_sincos_pos_embed


if __name__ == '__main__':
    IMAGE_SIZE = 224
    model = FCViT()  # turn off shuffling process

    # FCViT
    device = 'cpu'
    batch_size = 8
    num_batches = 30
    inputs = torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)

    start_time = time.time()
    with torch.no_grad():
        for i in range(num_batches):
            _ = model(inputs)
    end_time = time.time()
    total_time = end_time - start_time
    inference_time = total_time / (batch_size * num_batches)
    print(f"inference_time: {inference_time:.10f} second")

    # JPDVT
    # device = 'cpu'
    # batch_size = 32
    # num_batches = 5
    # inputs = torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    # t = torch.full((batch_size,), 1000)
    #
    # from jpdvt_diffusion import create_diffusion
    # diffusion = create_diffusion('1000')
    # time_emb = torch.tensor(get_2d_sincos_pos_embed(8, 3)).unsqueeze(0).float().to(device)
    # time_emb = time_emb.repeat(batch_size, 1, 1).to(device)
    # time_emb_noise = torch.tensor(get_2d_sincos_pos_embed(8, 12)).unsqueeze(0).float().to(device)
    # time_emb_noise = time_emb_noise.repeat(batch_size, 1, 1)
    # time_emb_noise = torch.randn_like(time_emb_noise).to(device)
    # model_kwargs = None
    #
    # start_time = time.time()
    # with torch.no_grad():
    #     for i in range(num_batches):
    #         _ = model(inputs, t, time_emb)
    #         # samples = diffusion.p_sample_loop(
    #         #     model.forward, inputs, time_emb_noise.shape, time_emb_noise, clip_denoised=False,
    #         #     model_kwargs=model_kwargs, progress=True, device=device
    #         # )
    # end_time = time.time()
    # total_time = end_time - start_time
    # inference_time = total_time / (batch_size * num_batches)
    # print(f"inference_time: {inference_time:.10f} second")