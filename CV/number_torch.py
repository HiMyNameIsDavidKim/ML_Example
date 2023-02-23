import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchsummary import summary


batch_size = 256
learning_rate = 0.001
num_epoch = 5
device = 'mps'
model_path = './save/number_CNN.pt'


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 16, 5),  # [batch_size,1,28,28] -> [batch_size,16,24,24]
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),  # [batch_size,16,24,24] -> [batch_size,32,20,20]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [batch_size,32,20,20] -> [batch_size,32,10,10]
            nn.Conv2d(32, 64, 5),  # [batch_size,32,10,10] -> [batch_size,64,6,6]
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # [batch_size,64,6,6] -> [batch_size,64,3,3]
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 3 * 3, 100),  # [batch_size,64*3*3] -> [batch_size,100]
            nn.ReLU(),
            nn.Linear(100, 10)  # [batch_size,100] -> [batch_size,10]
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        return out


class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.output = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x


class NumberModel(object):
    def __init__(self):
        self.mnist_train = None
        self.mnist_test = None
        self.train_loader = None
        self.test_loader = None
        self.model = None

    def process(self):
        self.dataset()
        self.modeling()
        self.save_model()

    def dataset(self):
        mnist_train = dset.MNIST(root="./data/", train=True,
                                 transform=transforms.ToTensor(),
                                 target_transform=None,
                                 download=True)
        mnist_test = dset.MNIST(root="./data/", train=False,
                                transform=transforms.ToTensor(),
                                target_transform=None,
                                download=True)
        print(mnist_train.__getitem__(0)[0].size(), mnist_train.__len__())
        print(mnist_test.__getitem__(0)[0].size(), mnist_test.__len__())
        print(len(mnist_train), len(mnist_test))
        self.mnist_train = mnist_train
        self.mnist_test = mnist_test

        self.train_loader = DataLoader(mnist_train,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=2,
                                       drop_last=True)
        self.test_loader = DataLoader(mnist_test,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=2,
                                      drop_last=True)

    def modeling(self):
        model = DNNModel().to(device)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        loss_arr = []
        for i in range(num_epoch):
            for j, [image, label] in enumerate(self.train_loader):
                x = image.to(device)
                y_ = label.to(device)

                optimizer.zero_grad()
                output = model.forward(x)
                loss = loss_func(output, y_)
                loss.backward()
                optimizer.step()

                if j % 1000 == 0:
                    print(loss)
                    loss_arr.append(loss.cpu().detach().numpy())

        self.model = model

    def save_model(self):
        torch.save(self.model, model_path)

    def eval_test(self):
        self.dataset()
        model = torch.load(model_path)
        correct = 0
        total = 0

        with torch.no_grad():
            for image, label in self.test_loader:
                x = image.to(device)
                y_ = label.to(device)

                output = model.forward(x)
                _, output_index = torch.max(output, 1)

                total += label.size(0)
                correct += (output_index == y_).sum().float()

            print(total)
            print(f"Accuracy of Test Data: {100 * correct / total:.2f}%")


class NumberService(object):
    def __init__(self):
        self.mnist_train = None
        self.mnist_test = None
        self.train_loader = None
        self.test_loader = None
        self.i = None
        self.labels = []
        self.preds = []

    def process(self):
        self.dataset()
        if len(self.labels) == 0 or len(self.preds) == 0:
            self.predict_first()
        self.idx()
        self.service_model()
        self.show()

    def dataset(self):
        mnist_train = dset.MNIST(root="./data/", train=True,
                                 transform=transforms.ToTensor(),
                                 target_transform=None,
                                 download=True)
        mnist_test = dset.MNIST(root="./data/", train=False,
                                transform=transforms.ToTensor(),
                                target_transform=None,
                                download=True)
        print(mnist_train.__getitem__(0)[0].size(), mnist_train.__len__())
        print(mnist_test.__getitem__(0)[0].size(), mnist_test.__len__())
        print(len(mnist_train), len(mnist_test))
        self.mnist_train = mnist_train
        self.mnist_test = mnist_test

        self.train_loader = DataLoader(mnist_train,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=2,
                                       drop_last=True)
        self.test_loader = DataLoader(mnist_test,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=2,
                                      drop_last=True)

    def idx(self):
        print(f'test number range : 0~{len(self.preds)}')
        self.i = int(input('input test number : '))

    def service_model(self):
        i = self.i
        print(f'Predict : {self.preds[i]}')
        print(f'Answer : {self.labels[i]}')

    def predict_first(self):
        model = torch.load(model_path)

        with torch.no_grad():
            for image, label in self.test_loader:
                x = image.to(device)
                y_ = label.to(device)

                output = model.forward(x)
                _, output_index = torch.max(output, 1)

                self.preds += list(output_index.to('cpu').numpy())
                self.labels += list(y_.to('cpu').numpy())

    def show(self):
        i = self.i
        image, label = self.mnist_test[i]
        plt.imshow(image.reshape(28, 28), cmap='gist_yarg')
        plt.show()


number_menus = ["Exit",  # 0
                "Modeling",  # 1
                "Evaluation Test",  # 2
                "Service",  # 3
                ]

number_lambda = {
    "1": lambda t: NumberModel().process(),
    "2": lambda t: NumberModel().eval_test(),
    "3": lambda t: t.process(),
    "4": lambda t: print(" ** No Function ** "),
    "5": lambda t: print(" ** No Function ** "),
    "6": lambda t: print(" ** No Function ** "),
    "7": lambda t: print(" ** No Function ** "),
    "8": lambda t: print(" ** No Function ** "),
    "9": lambda t: print(" ** No Function ** "),
}

if __name__ == '__main__':
    ns = NumberService()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(number_menus)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("### Exit ###")
            break
        else:
            try:
                number_lambda[menu](ns)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")
