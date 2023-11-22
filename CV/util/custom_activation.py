import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class MaskingLayer(nn.Module):
    def forward(self, x, masking=(10, 30)):
        (masking_low, masking_high) = masking
        return torch.where((x < masking_low) | (masking_high < x), x, torch.tensor(0, dtype=x.dtype).to(x.device))


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.masking_layer = MaskingLayer()

    def forward(self, x):
        x = self.masking_layer(x, masking=(0, 1))
        return x


if __name__ == '__main__':
    model = CustomModel()
    input_data = torch.randn(1, 256)
    output_data = model(input_data)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(input_data.numpy().flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Input Data Distribution')
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(output_data.detach().numpy().flatten(), bins=50, color='orange', alpha=0.7)
    plt.title('Model Output Distribution')
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()