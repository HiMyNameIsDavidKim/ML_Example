import torch
from matplotlib import pyplot as plt


def acc_compare(self, *paths):
    plt.figure(figsize=(4, 3), dpi=200)
    for i, path in enumerate(paths):
        checkpoint = torch.load(path, map_location='cpu')
        self.epochs = checkpoint['epochs']
        self.acc = checkpoint['accuracies']
        ls_x = []
        ls_y = []
        for x, y in enumerate(self.acc):
            ls_x.append(x)
            ls_y.append(y)
        plt.plot(ls_x, ls_y, label=f'Curve {i+1}')
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Validation Accuracy', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend()
    plt.show()