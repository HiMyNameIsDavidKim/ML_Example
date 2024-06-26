# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block
from torchsummary import summary

from mae_util import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
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
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Jigsaw decoder specifics
        self.jigsaw_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.num_patches = num_patches

        self.jigsaw_decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.jigsaw_decoder_norm = norm_layer(decoder_embed_dim)
        self.jigsaw_decoder_pred = nn.Linear(decoder_embed_dim, num_patches, bias=True)

        self.dic_fix = {}
        self.fix_encoder = True
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        self.mode_of_jigsaw_train()

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

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_jigsaw_decoder(self, x, ids_restore, threshold):
        # embed tokens
        x = self.jigsaw_decoder_embed(x)

        # append mask tokens to sequence
        len_keep = x.shape[1]
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - len_keep, 1)
        # x = torch.cat([x, mask_tokens], dim=1)
        # test point
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # apply Transformer blocks
        for blk in self.jigsaw_decoder_blocks:
            x = blk(x)
        x = self.jigsaw_decoder_norm(x)

        # predictor projection
        x = self.jigsaw_decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        # sort for debug
        ids_pred = torch.argmax(x, dim=2)  # [n, 196], int, 각 패치의 위치 class 예측값 ids
        ids_temp = torch.cat((ids_pred[:, :len_keep], ids_restore[:, len_keep:]), dim=1)  # [언마스킹 예측 + 마스킹 정답]
        ids_temp = torch.argsort(ids_temp, dim=1)  # 오류 방지를 위해 다시 한번 sorting

        x = x[:, :len_keep, :]
        target_jigsaw = ids_restore[:, :len_keep]

        # eval acc
        if self.fix_encoder:
            self.eval_jigsaw_acc(x, target_jigsaw, threshold)

        return x, ids_temp, target_jigsaw

    def eval_jigsaw_acc(self, pred_jigsaw, target_jigsaw, threshold):
        pred_jigsaw = torch.argmax(pred_jigsaw, dim=2)
        total = target_jigsaw.size(0) * target_jigsaw.size(1)
        correct = (pred_jigsaw == target_jigsaw).sum().item()
        acc = correct / total
        if acc > threshold:
            self.mode_of_encoder_train()
            self.fix_encoder = False

    def mode_of_jigsaw_train(self):
        for name, param in self.named_parameters():
            self.dic_fix[name] = param.requires_grad
            param.requires_grad = False
            if 'jigsaw_decoder' in name:
                param.requires_grad = True

    def mode_of_encoder_train(self):
        for name, param in self.named_parameters():
            if name in self.dic_fix:
                param.requires_grad = self.dic_fix[name]
            if 'decoder' in name:
                param.requires_grad = False

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, pred_jigsaw, target_jigsaw):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss_recon = (pred - target) ** 2
        loss_recon = loss_recon.mean(dim=-1)  # [N, L], mean loss per patch
        loss_recon = (loss_recon * mask).sum() / mask.sum()  # mean loss on removed patches

        # reshape for input to cross_entropy (n * 49, 196) and (n* 49, )
        weight_ratio = 0.01
        _, _, L = pred_jigsaw.shape
        loss_jigsaw = F.cross_entropy(pred_jigsaw.reshape(-1, L), target_jigsaw.reshape(-1))

        loss = loss_recon + loss_jigsaw * weight_ratio

        if self.fix_encoder:
            return loss_jigsaw
        else:
            return loss

    def forward(self, imgs, mask_ratio=0.75, threshold=0.83):
        """
        data type
        imgs: [n, 3, 224, 224], 원본 이미지
        latent: [n, 50, 1024], 언마스킹인 애들에 대한 레이턴트 매트릭스, CLS 토큰 포함, 디멘션은 케바케
        mask: [n, 196], (0=언마스킹 49개, 1=마스킹 147개)
        ids_restore = [n, 196], int, 섞인 패치의 고유 ids
        pred_jigsaw: [n, 49, 196], prob, 언마스킹 패치들의 위치에 대한 representation
        target_jigsaw = ids_restore: [n, 49], int, 언마스킹 패치들의 ids 정답지
        ids_temp: [n, 196], int, pred_jigsaw와 target_jigsaw를 겹치는 값이 없도록 수정한 값
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred_jigsaw, ids_temp, target_jigsaw = self.forward_jigsaw_decoder(latent, ids_restore, threshold)
        pred_recon = self.forward_decoder(latent, ids_temp)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred_recon, mask, pred_jigsaw, target_jigsaw)

        # 테스트
        # 언셔플 해서 순서 맞추는지 보기 (진행중) (학습 못하는디....)
        # 언셔플+언마스킹 해서 원래 순서를 넣었을때 원래 순서를 알아야 하는데 모르면 의미가 없음.
        # 이러면 순서를 풀어내서 하는 법을 일단 알고리즘 속에 넣어야함.
        # 셔플 해서 순서 맞추는지 보기
        return loss, pred_recon, mask, pred_jigsaw, target_jigsaw


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == '__main__':
    # set recommended archs
    mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
    mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
    mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
