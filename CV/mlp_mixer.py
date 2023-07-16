import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_CLASSES = 10
NUM_LAYERS = 8
HIDDEN_FEATURES = 512
TOKENS_MLP_DIM = 256
CHANNELS_MLP_DIM = 2048

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_features, tokens_mlp_dim, channels_mlp_dim):
        super(MixerLayer, self).__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(hidden_features),
            nn.Linear(num_patches, tokens_mlp_dim),
            nn.GELU(),
            nn.Linear(tokens_mlp_dim, num_patches)
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(hidden_features),
            nn.Linear(hidden_features, channels_mlp_dim),
            nn.GELU(),
            nn.Linear(channels_mlp_dim, hidden_features)
        )

    def forward(self, x):
        y = x.permute(0, 2, 1)
        y = self.token_mixing(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.channel_mixing(x)
        x = x + y
        return x

class MLPMixer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_layers, hidden_features, tokens_mlp_dim, channels_mlp_dim):
        super(MLPMixer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(3, hidden_features, kernel_size=patch_size, stride=patch_size)
        self.mixer_layers = nn.ModuleList([
            MixerLayer(num_patches, hidden_features, tokens_mlp_dim, channels_mlp_dim)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_features)
        self.fc = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = MLPMixer(image_size=IMAGE_SIZE,
                     patch_size=PATCH_SIZE,
                     num_classes=NUM_CLASSES,
                     num_layers=NUM_LAYERS,
                     hidden_features=HIDDEN_FEATURES,
                     tokens_mlp_dim=TOKENS_MLP_DIM,
                     channels_mlp_dim=CHANNELS_MLP_DIM,
                     )