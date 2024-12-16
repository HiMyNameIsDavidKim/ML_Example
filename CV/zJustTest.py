import torch
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis
from typing import Tuple, DefaultDict, Counter, Dict
from fvcore.nn.jit_handles import Handle
from collections import defaultdict

from puzzle_fcvit import FCViT


if __name__ == '__main__':
    model = FCViT()
    output, target = model(torch.rand(2, 3, 225, 225))
    summary(model, (3, 225, 225))

    input_tensor = torch.randn(1, 3, 225, 225)
    flops = FlopCountAnalysis(model, input_tensor)
    print(f"\nTotal FLOPs: {flops.total() / 1e9:.6f} GFLOPs")

