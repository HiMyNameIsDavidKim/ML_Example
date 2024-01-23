from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


class SPRTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        super(SPRTransformer, self).__init__(**kwargs)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        outcome = x[:, 0]

        return outcome


def sprt_base_patch16_img192(**kwargs):
    model = SPRTransformer(
        img_size=192, patch_size=16, in_chans=6, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
