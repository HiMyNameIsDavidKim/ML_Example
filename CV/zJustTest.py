import torch
import pandas as pd
from matplotlib import pyplot as plt


def tracker(file_name):
    df = pd.read_csv(file_name, header=None)

    tensors = []
    for i in range(0, len(df.columns), 3):
        if i + 1 < len(df.columns):
            tensor = torch.tensor(df.iloc[:, [i, i + 1]].values, dtype=torch.float32)
            tensors.append(tensor)

    label = tensors[0]
    center = torch.ones(9, 2)
    corrects = []
    center_preds = []

    for idx, pred in enumerate(tensors[1:]):
        correct = (pred == label).all(dim=1).sum().item()
        center_pred = (pred == center).all(dim=1).sum().item()
        corrects.append(correct)
        center_preds.append(center_pred)

    plt.figure(figsize=(10, 5))
    epochs = range(len(corrects))

    plt.plot(epochs, center_preds, label='Center', marker='s')
    plt.plot(epochs, corrects, label='Correct', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.title('Tracking')
    plt.xticks(epochs)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    file_name = f'./data/tracking_sample_{6}.csv'
    tracker(file_name)
