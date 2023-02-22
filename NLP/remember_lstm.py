import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RememberLSTMModel(nn.Module):
    def __init__(self):
        super(RememberLSTMModel, self).__init__()

        global n_hidden, lr, epochs, string, chars, char_list, n_letters, device, model_path, \
            batch_size, seq_len, num_layers
        n_hidden = 35
        lr = 0.01
        epochs = 1000
        string = "hello pytorch. how long can a rnn cell remember? show me your limit!"
        chars = "abcdefghijklmnopqrstuvwxyz ?!.,:;01"
        char_list = [i for i in chars]
        n_letters = len(char_list)
        device = 'cpu'
        model_path = './save/remember_LSTM.pt'
        batch_size = 1
        seq_len = 1
        num_layers = 3
        self.input_size = None
        self.hidden_size = None
        self.num_layers = None
        self.lstm = None
        self.output_size = None
        self.i2h = None
        self.i2o = None
        self.act_fn = None
        self.model = None


    def process(self):
        self.architect()
        self.modeling()
        self.save_model()

    def string_to_onehot(self, string):
        start = np.zeros(shape=n_letters, dtype=int)
        end = np.zeros(shape=n_letters, dtype=int)
        start[-2] = 1
        end[-1] = 1
        for i in string:
            idx = char_list.index(i)
            zero = np.zeros(shape=n_letters, dtype=int)
            zero[idx] = 1
            start = np.vstack([start, zero])
        output = np.vstack([start, end])
        return output

    def onehot_to_word(self, onehot_1):
        onehot = torch.Tensor.numpy(onehot_1)
        return char_list[onehot.argmax()]

    def architect(self):
        self.input_size = n_letters
        self.hidden_size = n_hidden
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)

    def forward(self, input_, hidden, cell):
        output, (hidden, cell) = self.lstm(input_, (hidden, cell))
        return output, hidden, cell

    def init_hidden_cell(self):
        hidden = torch.zeros(num_layers, batch_size, n_hidden)
        cell = torch.zeros(num_layers, batch_size, n_hidden)
        return hidden, cell

    def modeling(self):
        model = self.to(device)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        one_hot = torch.from_numpy(self.string_to_onehot(string)).type_as(torch.FloatTensor())

        j = 0
        input_data = one_hot[j:j + seq_len].view(seq_len, batch_size, n_letters)

        unroll_len = one_hot.size()[0] // seq_len - 1
        for i in range(epochs):
            hidden, cell = model.init_hidden_cell()

            loss = 0
            for j in range(unroll_len):
                input_data = one_hot[j:j + seq_len].view(seq_len, batch_size, n_letters)
                label = one_hot[j + 1:j + seq_len + 1].view(seq_len, batch_size, n_letters)

                optimizer.zero_grad()

                output, hidden, cell = model(input_data, hidden, cell)
                loss += loss_func(output.view(1, -1), label.view(1, -1))

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(loss)


        self.model = model

    def save_model(self):
        torch.save(self.model, model_path)

    def eval_test(self):
        self.architect()
        model = torch.load(model_path)
        hidden, cell = model.init_hidden_cell()
        output_string = ""

        one_hot = torch.from_numpy(self.string_to_onehot(string)).type_as(torch.FloatTensor())

        unroll_len = one_hot.size()[0] // seq_len - 1
        for j in range(unroll_len - 1):
            input_data = one_hot[j:j + 1].view(1, batch_size, n_letters)
            label = one_hot[j + 1:j + 1 + 1].view(1, batch_size, n_letters)

            output, hidden, cell = model(input_data, hidden, cell)

            output_string += self.onehot_to_word(output.data)

        print(f'[Remember]\n{output_string}')
        print(f'[Answer]\n{string}')

if __name__ == '__main__':
    # RememberLSTMModel().process()
    RememberLSTMModel().eval_test()