import torch
import torch.nn as nn
import unidecode
import string
import random
import re
import time, math


class ShakeSpeareGRUModel(nn.Module):
    def __init__(self):
        super(ShakeSpeareGRUModel, self).__init__()

        global num_epochs, print_every, plot_every, chunk_len, \
            hidden_size, batch_size, num_layers, embedding_size, lr, \
            all_characters, n_characters, file, file_len, device, model_path
        num_epochs = 2000
        print_every = 100
        plot_every = 10
        chunk_len = 200
        hidden_size = 100
        batch_size = 1
        num_layers = 2
        embedding_size = 70
        lr = 0.002
        all_characters = string.printable
        n_characters = len(all_characters)
        file = unidecode.unidecode(open('./data/shakespeare.txt').read())
        file_len = len(file)
        device = 'cpu'
        model_path = './save/shakespeare_GRU.pt'
        self.input_size = None
        self.embedding_size = None
        self.hidden_size = None
        self.output_size = None
        self.num_layers = None
        self.encoder = None
        self.rnn = None
        self.decoder = None
        self.model = None

    def process(self):
        self.architect()
        self.modeling()
        self.save_model()

    def random_chunk(self):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        return file[start_index:end_index]

    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = all_characters.index(string[c])
        return tensor

    def random_training_set(self):
        chunk = self.random_chunk()
        inp = self.char_tensor(chunk[:-1])
        target = self.char_tensor(chunk[1:])
        return inp, target, chunk

    def test(self, model):
        start_str = "b"
        inp = self.char_tensor(start_str)
        hidden = model.init_hidden()
        x = inp

        print(start_str, end="")
        for i in range(200):
            output, hidden = model(x, hidden)

            output_dist = output.data.view(-1).div(0.8).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            predicted_char = all_characters[top_i]

            print(predicted_char, end="")

            x = self.char_tensor(predicted_char)

    def architect(self):
        self.input_size = n_characters
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = n_characters
        self.num_layers = num_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputt, hidden):
        out = self.encoder(inputt.view(1, -1))
        out, hidden = self.rnn(out, hidden)
        out = self.decoder(out.view(batch_size, -1))
        return out, hidden

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden

    def modeling(self):
        model = self.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_func = nn.CrossEntropyLoss()

        for i in range(num_epochs):
            inp, label, chunk = self.random_training_set()
            hidden = model.init_hidden()

            loss = torch.tensor([0]).type(torch.FloatTensor)
            optimizer.zero_grad()
            for j in range(chunk_len - 1):
                x = inp[j]
                y_ = label[j].unsqueeze(0).type(torch.LongTensor)
                y, hidden = model(x, hidden)
                loss += loss_func(y, y_)

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("\n", loss / chunk_len, "\n")
                print("\n", "=" * 50, 'target', "=" * 50)
                print(chunk)
                print("\n", "=" * 50, 'remember', "=" * 50)
                self.test(model)
                print("\n", "=" * 100)

        self.model = model

    def save_model(self):
        torch.save(self.model, model_path)


if __name__ == '__main__':
    ShakeSpeareGRUModel().process()
