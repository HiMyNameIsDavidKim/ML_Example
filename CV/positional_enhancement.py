import timm
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.optim import AdamW, SGD
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsummary import summary
import torchvision.models as models


class PositionalEnhanceViT(nn.Module):
    def __init__(self, num_classes):
        super(PositionalEnhanceViT, self).__init__()

        self.vit_origin = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
        self.vit_origin.num_classes = num_classes

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
        x = torch.cat([x, self.pos_embed], dim=2)
        x = self.norm1(x)
        x = self.fc2(self.act(self.fc1(x)))
        x = self.norm2(x)
        x = self.vit_origin(x)
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