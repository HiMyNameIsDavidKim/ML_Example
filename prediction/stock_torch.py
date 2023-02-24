import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.nn.functional as F


start = datetime(2021, 1, 4)
end = datetime(2023, 1, 2)
name = '005930.KS'
ticker = yf.Ticker(name)
device = 'cpu'
num_epochs = 30000
lr = 0.01
num_classes = 1
input_size = 3
hidden_size = 3
num_layers = 1
seq_length = 4
model_path = './save/stock_LSTM_Ensemble.pt'


'''
Open            float64
High            float64
Low             float64
Close           float64
Adj Close       float64
Volume            int64
Dividends       float64
Stock Splits    float64
'''


class LSTMEnModel(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTMEnModel, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = F.relu(hn)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class StockModel(object):
    def __init__(self):
        self.samsung = ticker.history(interval='1d', start=start, end=end, auto_adjust=False)
        self.samsung = self.samsung.drop(columns=['Adj Close', 'Volume', 'Dividends', 'Stock Splits'])
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.hist = None

    def process(self):
        self.prepare_ds()
        self.train_model()
        self.loss_plot()
        self.save_model()

    # def split_xy(self, dataset):
    #     dataset = np.array(dataset.values)
    #     x, y = [], []
    #     for i in range(len(dataset)):
    #         x_end_number = i + 14
    #         y_end_number = x_end_number + 1
    #
    #         if y_end_number > len(dataset)-1:
    #             break
    #         tmp_x = dataset[i:x_end_number, :]
    #         tmp_y = dataset[x_end_number:y_end_number, 3]
    #         x.append(tmp_x)
    #         y.append(tmp_y)
    #     return np.array(x), np.array(y)

    def prepare_ds(self):
        x = self.samsung.drop(columns=['Close'])
        y = self.samsung.iloc[:, 3:4]
        ratio = int(x.shape[0] * 0.80)

        scaler_x = StandardScaler()
        scaler_x.fit(x)
        x = scaler_x.transform(x)

        x_train = x[:ratio, :]
        x_test = x[ratio:, :]
        y_train = np.array(y)[:ratio, :]
        y_test = np.array(y)[ratio:, :]

        x_train = Variable(torch.Tensor(x_train))
        x_test = Variable(torch.Tensor(x_test))
        self.y_train = Variable(torch.Tensor(y_train))
        self.y_test = Variable(torch.Tensor(y_test))

        self.x_train = torch.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        self.x_test = torch.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

        print("Training Shape", x_train.shape, y_train.shape)
        print("Testing Shape", x_test.shape, y_test.shape)

    def train_model(self):
        model = LSTMEnModel(num_classes, input_size, hidden_size, num_layers, self.x_train.shape[1]).to(device)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.hist = np.zeros(num_epochs)
        for epoch in range(num_epochs):
            x = self.x_train.to(device)
            y_ = self.y_train.to(device)

            optimizer.zero_grad()

            outputs = model.forward(x)
            loss = loss_function(outputs, y_)

            loss.backward()
            optimizer.step()

            if epoch % 300 == 0:
                print(f"Epoch: {epoch}, loss: {loss.item():.5f}")
            self.hist[epoch] = loss.item()

        self.model = model

    def loss_plot(self):
        x = [self.hist[i] for i in range(len(self.hist)) if i % 300 == 0]
        plt.plot(x, label='loss')
        plt.legend()
        plt.show()

    def save_model(self):
        torch.save(self.model, model_path)

    def eval_test(self):
        model = torch.load(model_path)
        x = self.samsung.drop(columns=['Close'])
        y = self.samsung.iloc[:, 3:4]

        scaler_x = StandardScaler()
        scaler_x.fit(x)
        x = scaler_x.transform(x)

        x_eval = Variable(torch.Tensor(np.array(x)))
        y_eval = Variable(torch.Tensor(np.array(y)))
        x_eval = torch.reshape(x_eval, (x_eval.shape[0], 1, x_eval.shape[1]))

        train_predict = model(x_eval.to(device))
        data_predict = train_predict.data.detach().cpu().numpy()
        dataY_plot = y_eval.data.numpy()

        plt.figure(figsize=(10, 6))
        plt.axvline(x=x_eval.shape[0], c='r', linestyle='--')
        plt.plot(dataY_plot, label='Actuall Data')
        plt.plot(data_predict, label='Predicted Data')
        plt.title('Time-Series Prediction')
        plt.legend()
        plt.show()



if __name__ == '__main__':
    StockModel().process()
    StockModel().eval_test()
