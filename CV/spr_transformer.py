from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed, Block

from sprt_util import get_2d_sincos_pos_embed


# --------------------------------------------------------
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------
# SPRNN
# img_size=192, patch_size=16, in_chans=2
# input = [batch, 2, 192, 192]
# dim = [batch, 256, 192, 192]
# MaxPool2d 1/2
# output = [batch 1, 96, 96]
# --------------------------------------------------------
# SPRT
# img_size=192, patch_size=16, in_chans=2
# input = [batch, 2, 192, 192]
# number of patch = (192 / 16)^2 = 12*12
# encoder_dim = [batch, 12*12, 512]
# decoder_dim = [batch, 12*12, 512]  # lower?
# decoder_output = [batch, 12*12, 64], [batch, 12*12, 32]
# final_output = [batch, 1, 96, 96], [batch, 1, 96, 48]
# --------------------------------------------------------


class SPRTransformer(nn.Module):
    def __init__(self, img_size=192, patch_size=16, in_chans=2, embed_dim=512, depth=12, num_heads=12,
                 out_patch_size=8, out_chans=1, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        # --------------------------------------------------------------------------
        # encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred_g = nn.Linear(decoder_embed_dim, out_patch_size ** 2 * out_chans, bias=True)  # decoder to patch
        self.decoder_pred_rb = nn.Linear(decoder_embed_dim, (out_patch_size ** 2 * out_chans)/2, bias=True)  # decoder to patch
        self.out_patch_size = out_patch_size
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_decoder(self, x, pattern):
        x = self.decoder_embed(x)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        if pattern == 'green':
            x = self.decoder_pred_g(x)
        else:
            x = self.decoder_pred_rb(x)
        x = x[:, 1:, :]
        return x

    def unpatchify(self, x, pattern):
        p = self.out_patch_size  # 8
        h = w = int(x.shape[1] ** .5)  # 12
        assert h * w == x.shape[1]

        if pattern == 'green':
            x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
            x = torch.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
            return imgs
        else:
            x = x.reshape(shape=(x.shape[0], h, w, p, p/2, 1))
            x = torch.einsum('nhwpqc->nchpwq', x)
            imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p/2))
            return imgs

    def forward(self, x, pattern='green'):
        x = self.forward_encoder(x)
        x = self.forward_decoder(x, pattern)
        x = self.unpatchify(x, pattern)
        return x


def sprt_base_patch16_img192(**kwargs):
    model = SPRTransformer(
        img_size=192, patch_size=16, in_chans=2, embed_dim=512, depth=12, num_heads=12,
        out_patch_size=8, out_chans=1, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

