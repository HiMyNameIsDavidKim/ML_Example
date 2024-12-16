import torch
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis
from typing import Tuple, DefaultDict, Counter, Dict
from fvcore.nn.jit_handles import Handle
from collections import defaultdict

from puzzle_fcvit import FCViT
from puzzle_cfn import PuzzleCFN
from puzzle_jpdvt import JPDVT


if __name__ == '__main__':
    # model = FCViT()
    # input_tensor = torch.randn(1, 3, 225, 225)
    # flops = FlopCountAnalysis(model, input_tensor)

    # model = PuzzleCFN(classes=1000)
    # input_tensor = torch.randn(1, 9, 3, 75, 75)
    # flops = FlopCountAnalysis(model, input_tensor)

    IMAGE_SIZE = 192
    model = JPDVT(input_size=IMAGE_SIZE)
    inputs = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    t = torch.full((2,), 1000)
    time_emb = torch.rand(2, 144, 8)
    outputs = model(inputs, t, time_emb)  # x, t, time_emb
    flops = FlopCountAnalysis(model, inputs=(inputs, t, time_emb))

    print(f"\nTotal FLOPs: {flops.total() / 1e9:.6f} GFLOPs")

