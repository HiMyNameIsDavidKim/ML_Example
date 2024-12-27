import time

import torch

from puzzle_fcvit_4x4 import FCViT
from puzzle_cfn import PuzzleCFN
from puzzle_jpdvt import JPDVT, get_2d_sincos_pos_embed


if __name__ == '__main__':
    IMAGE_SIZE = 224
    model = FCViT()  # turn off shuffling process

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

