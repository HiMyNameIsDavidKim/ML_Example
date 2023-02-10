import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RememberModel(nn.Module):
    def __init__(self):
        super(RememberModel, self).__init__()

        global n_hidden, lr, epochs, string, chars, char_list, n_letters, device, model_path
        n_hidden = 35
        lr = 0.01
        epochs = 1000
        string = "hello pytorch. how long can a rnn cell remember? show me your limit!"
        chars = "abcdefghijklmnopqrstuvwxyz ?!.,:;01"
        char_list = [i for i in chars]
        n_letters = len(char_list)
        device = 'cpu'
        model_path = './save/remember_RNN.pt'
        self.input_size = None
        self.hidden_size = None
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
        self.output_size = n_letters
        self.i2h = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.i2o = nn.Linear(self.input_size + self.hidden_size, self.output_size)
        self.act_fn = nn.Tanh()

    def forward(self, inputt, hidden):
        combined = torch.cat((inputt, hidden), 1)
        hidden = self.act_fn(self.i2h(combined))
        output = self.i2o(combined)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def modeling(self):
        model = self.to(device)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        one_hot = torch.from_numpy(self.string_to_onehot(string)).type_as(torch.FloatTensor())

        for i in range(epochs):
            optimizer.zero_grad()
            hidden = model.init_hidden()

            total_loss = 0
            for j in range(one_hot.size()[0] - 1):
                input_ = one_hot[j:j + 1, :]
                target = one_hot[j + 1]
                output, hidden = model.forward(input_, hidden)

                loss = loss_func(output.view(-1), target.view(-1))
                total_loss += loss

            total_loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(total_loss)

        self.model = model

    def save_model(self):
        torch.save(self.model, model_path)

    def eval_test(self):
        self.architect()
        model = torch.load(model_path)
        start = torch.zeros(1, n_letters)
        start[:, -2] = 1

        with torch.no_grad():
            hidden = model.init_hidden()
            input_ = start
            output_string = ""

            for i in range(len(string)):
                output, hidden = model.forward(input_, hidden)
                output_string += self.onehot_to_word(output.data)
                input_ = output

        print(output_string)


if __name__ == '__main__':
    RememberModel().process()
    RememberModel().eval_test()