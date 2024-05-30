import PIL
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pyhessian import hessian # Hessian computation
from pytorchcv.model_provider import get_model as ptcv_get_model # model
from puzzle_vit import PuzzleViT

import matplotlib.pyplot as plt

BATCH_SIZE = 1
test_model_path = './save/puzzle_imagenet_1000_vitPreFalse_ep100_lr1e-05_b64.pt'

model = PuzzleViT()
# checkpoint = torch.load(test_model_path)
# model.load_state_dict(checkpoint['model'])
model.eval()
criterion = torch.nn.SmoothL1Loss()

transform = transforms.Compose([
    transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.Pad(padding=(0, 0, 1, 1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = datasets.ImageFolder('./data/ImageNet/val', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

print('### inputs and targets ###')
inputs, targets, outputs = None, None, None
for batch, (inputs, _) in enumerate(train_loader):
    inputs = inputs
    outputs, targets, loss_var = model(inputs)
    break

print('### hessian_comp ###')
hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=False)
top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)

lams1 = np.linspace(-0.5, 0.5, 21).astype(np.float32)
lams2 = np.linspace(-0.5, 0.5, 21).astype(np.float32)

loss_list = []

model_perb1 = PuzzleViT()
# checkpoint = torch.load(test_model_path)
# model_perb1.load_state_dict(checkpoint['model'])
model_perb1.eval()

model_perb2 = PuzzleViT()
# checkpoint = torch.load(test_model_path)
# model_perb2.load_state_dict(checkpoint['model'])
model_perb2.eval()

def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

print('### lam ###')
for lam1 in lams1:
    for lam2 in lams2:
        model_perb1 = get_params(model, model_perb1, top_eigenvector[0], lam1)
        model_perb2 = get_params(model_perb1, model_perb2, top_eigenvector[1], lam2)
        loss_list.append((lam1, lam2, criterion(model_perb2(inputs), targets).item()))

loss_list = np.array(loss_list)

for i in [30, 120, 210, 300]:
    fig = plt.figure()
    landscape = fig.gca(projection='3d')
    landscape.plot_trisurf(loss_list[:, 0], loss_list[:, 1], loss_list[:, 2], alpha=0.8, cmap='viridis')

    landscape.set_title('Loss Landscape')
    landscape.set_xlabel('ε_1')
    landscape.set_ylabel('ε_2')
    landscape.set_zlabel('Loss')

    landscape.view_init(elev=30, azim=i)
    landscape.dist = 8
    plt.show()