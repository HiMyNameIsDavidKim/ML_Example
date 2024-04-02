import torch
from matplotlib import pyplot as plt


def visualLossAcc(losses, accuracies, lim_loss=(0.5, 7), lim_acc=(0, 90)):
    fig, ax1 = plt.subplots()

    ax1.plot(losses, 'b', label='Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_ylim(lim_loss)
    ax1.set_title('Training Losses and Accuracies')

    ax2 = ax1.twinx()
    ax2.plot(accuracies, 'r', label='Accuracy')
    ax2.set_ylabel('Accuracy', color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_ylim(lim_acc)

    plt.xticks(range(0, len(losses), 20), range(0, len(losses)//20))

    plt.show()


def visualLoss(ls_loss1):
    fig, axs = plt.subplots(1, 1, figsize=(12, 4))
    axs[0].plot(range(len(ls_loss1)), ls_loss1, label='Loss')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Loss')

    plt.tight_layout()
    plt.show()


def visualDoubleLoss(ls_loss1, ls_loss2):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(range(len(ls_loss1)), ls_loss1, label='Coord Loss')
    axs[0].set_title('Coord Loss')
    axs[0].set_xlabel('Steps')
    axs[0].set_ylabel('Loss')

    axs[1].plot(range(len(ls_loss2)), ls_loss2, label='Total Loss')
    axs[1].set_title('Total Loss')
    axs[1].set_xlabel('Steps')
    axs[1].set_ylabel('Loss')

    plt.tight_layout()
    plt.show()


def visualMultiLoss(*ls_models):
    ls_losses = []
    for model in ls_models:
        checkpoint = torch.load(model)
        loss = checkpoint['losses']
        ls_losses.append(loss)

    plt.figure()
    lim_loss = (0.5, 7)

    for loss, name in zip(ls_losses, ls_models):
        plt.plot(loss, label=f'{name[7:-3]}')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tick_params('y')
    plt.ylim(lim_loss)
    plt.title('Training Multi Losses')
    plt.legend()

    plt.xticks(range(0, len(ls_losses[0]), 20), range(0, len(ls_losses[0]) // 20))

    plt.show()


if __name__ == '__main__':
    model_1 = [1,1,1]
    model_2 = [2,2,2]
    model_3 = [3,3,3]
    visualMultiLoss(model_1, model_2, model_3)