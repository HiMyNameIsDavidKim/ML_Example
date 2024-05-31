import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pyhessian import hessian # Hessian computation
from pytorchcv.model_provider import get_model as ptcv_get_model # model

import matplotlib.pyplot as plt


def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb


batch_size = 16
model = ptcv_get_model("resnet20_cifar10", pretrained=True)
model.eval()
criterion = torch.nn.CrossEntropyLoss()


transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

inputs, targets = None, None
for batch, (inputs, targets) in enumerate(train_loader):
    break


hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=False)  #cuda=True
top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)


lams1 = np.linspace(-0.5, 0.5, 21).astype(np.float32)
lams2 = np.linspace(-0.5, 0.5, 21).astype(np.float32)
loss_list = []

model_perb1 = ptcv_get_model("resnet20_cifar10", pretrained=True)
model_perb1.eval()
# model_perb1 = model_perb1.cuda()
model_perb2 = ptcv_get_model("resnet20_cifar10", pretrained=True)
model_perb2.eval()
# model_perb2 = model_perb2.cuda()

for lam1 in lams1:
    for lam2 in lams2:
        model_perb1 = get_params(model, model_perb1, top_eigenvector[0], lam1)
        model_perb2 = get_params(model_perb1, model_perb2, top_eigenvector[1], lam2)
        loss_list.append((lam1, lam2, criterion(model_perb2(inputs), targets).item()))
loss_list = np.array(loss_list)

# 3D plot
for i in [30, 120, 210, 300]:
    fig = plt.figure()
    landscape = Axes3D(fig)
    landscape.plot_trisurf(loss_list[:, 0], loss_list[:, 1], loss_list[:, 2], alpha=0.8, cmap='viridis')

    landscape.set_title('Loss Landscape')
    landscape.set_xlabel('ε_1')
    landscape.set_ylabel('ε_2')
    landscape.set_zlabel('Loss')

    landscape.view_init(elev=30, azim=i)
    landscape.dist = 8
    plt.show()

# 2D plot
loss_list = []
for lam in lams1:
    model_perb1 = get_params(model, model_perb1, top_eigenvector[0], lam)
    loss_list.append(criterion(model_perb1(inputs), targets).item())
loss_list = np.array(loss_list)

plt.plot(lams1, loss_list)
plt.ylabel('Loss')
plt.xlabel('Perturbation')
plt.title('Loss landscape perturbed based on top Hessian eigenvector')
plt.show()

loss_list = []
for lam in lams2:
    model_perb2 = get_params(model, model_perb2, top_eigenvector[1], lam)
    loss_list.append(criterion(model_perb2(inputs), targets).item())
loss_list = np.array(loss_list)

plt.plot(lams2, loss_list)
plt.ylabel('Loss')
plt.xlabel('Perturbation')
plt.title('Loss landscape perturbed based on top Hessian eigenvector')
plt.show()