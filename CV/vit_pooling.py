import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
image_size: 이미지의 크기 (224)
patch_size: 이미지를 패치로 나눌 크기 (16)
in_channels: 입력 이미지의 채널 수 (3)
num_classes: 분류해야 하는 클래스 수 (1000)
embed_dim: 패치 임베딩의 차원 (768)
depth: 인코더 블록의 수 (12)
num_heads: 멀티 헤드 어텐션의 헤드 수 (12)
qkv_bias: 어텐션의 행렬 연산에서 Q, K, V 행렬에 대한 바이어스 사용 여부 (기본값은 False)
'''


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEmbedding, self).__init__()
        position = torch.arange(0, num_patches, dtype=torch.float32)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pos_embedding = torch.zeros(1, num_patches, embed_dim)
        pos_embedding[0, :, 0::2] = torch.sin(position[:, None] * div_term[None, :embed_dim // 2])
        pos_embedding[0, :, 1::2] = torch.cos(position[:, None] * div_term[None, :embed_dim // 2])
        self.register_buffer('pos_embedding', pos_embedding)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x + self.pos_embedding
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = (q @ k.transpose(-2, -1)) * self.scale
        x = x.softmax(dim=-1)
        x = (x @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLPBody(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=True):
        super(MLPBody, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=False):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.mlp = MLPBody(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), out_features=embed_dim)
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class MLPHead(nn.Module):
    def __init__(self, embed_dim, mlp_hidden_dim, num_classes):
        super(MLPHead, self).__init__()
        self.embed_dim = embed_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(embed_dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.fc3 = nn.Linear(mlp_hidden_dim, num_classes)
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ViTPooling(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads):
        super(ViTPooling, self).__init__()
        self.patch_embed = PatchEmbedding(image_size=image_size, patch_size=patch_size, in_channels=in_channels,
                                          embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = PositionalEmbedding(num_patches=self.num_patches, embed_dim=embed_dim)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=False)
            for _ in range(depth)
        ])
        self.mlp_head = MLPHead(embed_dim=embed_dim, mlp_hidden_dim=embed_dim * 4, num_classes=num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        return x