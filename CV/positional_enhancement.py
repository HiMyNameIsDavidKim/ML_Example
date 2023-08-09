import timm
import torch
from torch import nn
import torchvision.models as models


class PositionalEnhanceViT(nn.Module):
    def __init__(self, num_classes):
        super(PositionalEnhanceViT, self).__init__()
        self.num_classes = num_classes
        self.vit_origin = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        self.vit_origin.num_classes = self.num_classes

        # patch > position embedding > concatenate > norm > MLP > norm > input
        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Parameter(torch.randn(1, 196, 768) * .02)
        self.norm1 = nn.LayerNorm(1536)
        self.fc1 = nn.Linear(1536, 768)
        self.fc2 = nn.Linear(768, 768)
        self.act = nn.GELU()
        self.norm2 = nn.LayerNorm(768)

    def forward(self, x):
        x = self.patch_embed(x)
        B, P, C = x.shape
        pos_embeds = torch.cat([self.pos_embed] * B, dim=0)
        x = torch.cat([x, pos_embeds], dim=2)
        x = self.norm1(x)
        x = self.fc2(self.act(self.fc1(x)))
        x = self.norm2(x)
        x = x.view(B, 3, 224, 224)
        x = self.vit_origin(x)
        return x


class PositionalEnhanceViTv2(nn.Module):
    def __init__(self, num_classes):
        super(PositionalEnhanceViTv2, self).__init__()
        self.num_classes = num_classes
        self.vit_origin = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        self.vit_origin.num_classes = self.num_classes
        self.features_vit = nn.Sequential(*list(self.vit_origin.children())[:-1])
        self.head_vit = nn.Sequential(*list(self.vit_origin.children())[-1:])

        self.patch_embed = PatchEmbed()
        self.pos_embed = nn.Parameter(torch.randn(1, 196, 768) * .02)
        self.norm1 = nn.LayerNorm(1536)
        self.fc1 = nn.Linear(1536, 768)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(768, 768)
        self.norm2 = nn.LayerNorm(768)

        self.norm3 = nn.LayerNorm(1536)
        self.fc3 = nn.Linear(1536, 768)
        self.act2 = nn.GELU()
        self.fc4 = nn.Linear(768, 768)
        self.norm4 = nn.LayerNorm(768)

        self.fc5 = nn.Linear(768, num_classes)

    def forward(self, x):
        x_vit = self.features_vit(x)
        x = self.patch_embed(x)
        B, P, C = x.shape
        pos_embeds = torch.cat([self.pos_embed] * B, dim=0)
        x = torch.cat([x, pos_embeds], dim=2)
        x = self.norm1(x)
        x = self.fc2(self.act1(self.fc1(x)))
        x = self.norm2(x)
        x = torch.cat([x, x_vit], dim=2)
        x = self.norm3(x)
        x = self.fc4(self.act2(self.fc3(x)))
        x = x[:, 0]
        x = self.norm4(x)
        x = self.fc5(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(
            self,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = (224, 224)
        patch_size = (16, 16)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class ModifiedResNet50(nn.Module):
    def __init__(self):
        super(ModifiedResNet50, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.new_layer = nn.Linear(3, 64)
        self.features = nn.Sequential(self.new_layer, *list(resnet.children())[1:])

    def forward(self, x):
        x = self.features(x)
        return x


if __name__ == '__main__':
    # model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
    # print(f'Parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # print(f'Classes: {model.num_classes}')
    # summary(model, input_size=(3, 224, 224))
    # dir(model)
    # help(model)

    # modified_resnet = ModifiedResNet50()
    # input_data = torch.randn(1, 3, 224, 224)
    # output = modified_resnet(input_data)
    # print(output)

    num_classes = 1000
    pe_vit = PositionalEnhanceViT(num_classes)
    input_data = torch.randn(1, 3, 224, 224)
    output = pe_vit(input_data)