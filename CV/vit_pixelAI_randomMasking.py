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
drop_rate: 포지션 임베딩과 인코더 블럭의 드롭 아웃 비율 설정 (0.1)
qkv_bias: 어텐션의 행렬 연산에서 Q, K, V 행렬에 대한 바이어스 사용 여부 (기본값은 True)
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
    def __init__(self, num_patches, embed_dim, drop_rate):
        super(PositionalEmbedding, self).__init__()
        position = torch.arange(0, num_patches, dtype=torch.float32)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pos_embedding = torch.zeros(1, num_patches, embed_dim)
        pos_embedding[0, :, 0::2] = torch.sin(position[:, None] * div_term[None, :embed_dim // 2])
        pos_embedding[0, :, 1::2] = torch.cos(position[:, None] * div_term[None, :embed_dim // 2])
        self.register_buffer('pos_embedding', pos_embedding)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = x + self.pos_embedding
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=True):
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
    def __init__(self, in_features, hidden_features, out_features, bias=True, drop_rate=0.):
        super(MLPBody, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.mlp = MLPBody(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio),
                           out_features=embed_dim, drop_rate=drop_rate)
        self.dropout = nn.Dropout(drop_rate)

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
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        return x


class ViTPixelAI(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, drop_rate=0.,
                 cls_token=None, change_mlp=False, mask_ratio=0.):
        super(ViTPixelAI, self).__init__()
        self.patch_embed = PatchEmbedding(image_size=image_size, patch_size=patch_size, in_channels=in_channels,
                                          embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) if cls_token is None else None
        self.pos_embed = PositionalEmbedding(num_patches=self.num_patches + 1, embed_dim=embed_dim, drop_rate=drop_rate)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=True, drop_rate=drop_rate)
            for _ in range(depth)
        ])
        self.mlp_head = MLPHead(embed_dim=embed_dim, mlp_hidden_dim=embed_dim * 4, num_classes=num_classes)
        self.change_mlp = change_mlp
        self.fc_1 = nn.Linear(embed_dim, 1000)
        self.fc_2 = nn.Linear(1000, 1000)
        self.fc_3 = nn.Linear(1000, 100)
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.full((1, 1, embed_dim), 0.5))  # 0.5로 구성된 마스킹 패치
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # torch.nn.init.normal_(self.mask_token, std=.02)  # 정규 분포로 구성된 마스킹 패치(0.5가 잘 안되면 사용)

    def forward_mlp(self, x):
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x

    def random_masking(self, x, mask_ratio):
        """
        x: [n, 196, 768], [batch, length, dim]
        x_masked: [n, 50, 768], [batch, masked length, dim]
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # masking 패치 선택을 위한 랜덤

        ids_shuffle = torch.argsort(noise, dim=1)  # small is 유지, large is 마스킹
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # 패치의 고유 id 저장

        ids_keep = ids_shuffle[:, :len_keep]  # 지정한 마스크 비율 만큼 id 리스트 컷팅
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # 패치들 컷팅

        mask_tokens = self.mask_token.repeat(x.shape[0], x.shape[1] + 1 - x_masked.shape[1], 1)  # 마스킹 토큰(=마스킹 패치)
        x = torch.cat([x_masked, mask_tokens], dim=1)  # 패치(컷팅된 일부) + 마스킹 패치
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_masked.shape[2]))  # unshuffle

        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_embed(x)
        x = self.random_masking(x, self.mask_ratio)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        x = x[:, 0]
        if self.change_mlp:
            x = self.forward_mlp(x)
            return x
        else:
            x = self.mlp_head(x)
            return x


if __name__ == '__main__':
    # Parameters
    IMAGE_SIZE = 27
    PATCH_SIZE = 3
    IN_CHANNELS = 14
    NUM_CLASSES = 1000
    EMBED_DIM = 126
    DEPTH = 9
    NUM_HEADS = 9
    DROP_RATE = 0.1
    CHANGE_MLP = True  # True 이면 MLP 변경
    MASK_RATIO = 0.75  # 마스킹 비율

    model = ViTPixelAI(image_size=IMAGE_SIZE,
                       patch_size=PATCH_SIZE,
                       in_channels=IN_CHANNELS,
                       num_classes=NUM_CLASSES,
                       embed_dim=EMBED_DIM,
                       depth=DEPTH,
                       num_heads=NUM_HEADS,
                       drop_rate=DROP_RATE,
                       change_mlp=CHANGE_MLP,
                       mask_ratio=MASK_RATIO,
                       )

    tensor_in = torch.rand(2, 14, 27, 27)
    tensor_out = model(tensor_in)
    print(f'input shape : {tensor_in.shape}')
    print(f'output shape : {tensor_out.shape}')
