import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
def inverse_transform(image_tensor):
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    inv_tensor = inv_normalize(image_tensor)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    return inv_tensor

def inout_images_plot(sample, pred):
    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(sample)
    axes[0].set_title("Input")
    axes[1].imshow(pred)
    axes[1].set_title("Output")
    [ax.axis('off') for ax in axes]

    plt.show()

