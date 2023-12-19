import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

class MaskingLayer(nn.Module):
    def __init__(self):
        super(MaskingLayer, self).__init__()
        scale_array = np.array(range(256), dtype=np.uint8)
        scale_image = Image.fromarray(scale_array)
        scale_tensor = transforms.ToTensor()(scale_image)
        self.scale_tensor_array = scale_tensor.numpy().flatten()

    def forward(self, x, masking=(10, 30)):
        (masking_low, masking_high) = masking
        masking_low = self.scale_tensor_array[masking_low]
        masking_high = self.scale_tensor_array[masking_high]
        return torch.where((x < masking_low) | (masking_high < x), x, torch.tensor(0, dtype=x.dtype).to(x.device))


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.masking_layer = MaskingLayer()

    def forward(self, x):
        x = self.masking_layer(x, masking=(10, 30))
        return x


if __name__ == '__main__':
    model = TestModel()
    input_data = torch.rand(size=(1, 100))
    output_data = model(input_data)

    print(output_data)

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