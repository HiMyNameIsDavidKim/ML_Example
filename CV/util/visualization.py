import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def inverse_transform(image_tensor):
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    inv_tensor = inv_normalize(image_tensor)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    return inv_tensor

def inout_images_plot(samples, mask, pred, model):
    pred = model.unpatchify(pred)
    pred = torch.einsum('nchw->nhwc', pred).detach().cpu()
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    samples = torch.einsum('nchw->nhwc', samples).cpu()
    img_masked = samples[0] * (1 - mask[0])
    img_paste = samples[0] * (1 - mask[0]) + pred[0] * mask[0]

    fig, axes = plt.subplots(1, 3)

    axes[0].imshow(torch.clip((samples[0] * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    axes[0].set_title("Input")
    axes[1].imshow(torch.clip((img_masked * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    axes[1].set_title("Masked")
    axes[2].imshow(torch.clip((img_paste * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    axes[2].set_title("Output")
    [ax.axis('off') for ax in axes]

    plt.tight_layout()
    plt.show()

def acc_jigsaw(pred_jigsaw, target_jigsaw):

    ## 다시 작성

    _, pred_jigsaw = torch.max(pred_jigsaw[0], 1)
    target_jigsaw = target_jigsaw[0]
    total = target_jigsaw.size(0)
    correct = (pred_jigsaw == target_jigsaw).sum().item()
    return correct, total