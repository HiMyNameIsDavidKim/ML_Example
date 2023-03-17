import numpy as np
import pandas as pd
from keras import Input, Model
from keras.optimizers import Adam
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, LSTM, concatenate, Dropout, LayerNormalization
from keras.saving.save import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

lotto_csv = './data/lotto_data.csv'
lotto_npy = './data/lotto_data.npy'
epochs = 10
col = 2


class LottoModel(object):
    def __init__(self):
        self.df_lotto = pd.read_csv(lotto_csv, index_col=0, header=0, encoding='utf-8', sep=',')
        self.df = None
        self.step = 0
        self.model = None
        self.path_model = None

    def process(self, l):
        self.df_modify()
        self.np_read()
        self.step = l
        self.path_model = f'./save/lotto/lotto_c{col}_t{l}.h5'
        x1, y1 = self.split_xy(self.df, self.step)
        x1_train, x1_test, y1_train, y1_test = self.dataset_lstm(x1, y1)
        self.build_model(x1_train, x1_test, y1_train, y1_test)
        self.predict_test(x1_test, y1_test)

    def df_modify(self):
        df = self.df_lotto
        df = df.drop(['bonus'], axis=1).dropna(axis=0)
        df = df.values
        np.save(lotto_npy, arr=df)

    def np_read(self):
        df = np.load(lotto_npy, allow_pickle=True)
        self.df = df

    def split_xy(self, df, time_steps):
        x, y = [], []
        for i in range(len(df)):
            x_end_number = i + time_steps
            y_end_number = x_end_number + 1

            if y_end_number > len(df):
                break
            tmp_x = df[i:x_end_number, :]
            tmp_y = df[x_end_number:y_end_number, col-1]
            x.append(tmp_x)
            y.append(tmp_y)
        return np.array(x), np.array(y)

    def dataset_lstm(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train).astype(float)
        x_test_scaled = scaler.transform(x_test).astype(float)
        x_train = np.reshape(x_train_scaled, (x_train_scaled.shape[0], self.step, 6)).astype(float)
        x_test = np.reshape(x_test_scaled, (x_test_scaled.shape[0], self.step, 6)).astype(float)
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)
        return x_train, x_test, y_train, y_test

    def build_model(self, x_train, x_test, y_train, y_test):
        input1 = Input(shape=(self.step, 6))
        dense1 = LSTM(64)(input1)
        dense1 = LayerNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)
        dense1 = Dense(32)(dense1)
        dense1 = LayerNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)
        output1 = Dense(1)(dense1)

        model = Model(inputs=input1, outputs=output1)
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mse'])

        early_stopping = EarlyStopping(patience=20)
        model.fit(x_train, y_train,
                  validation_split=0.2, verbose=1, batch_size=1, epochs=epochs,
                  callbacks=[early_stopping])
        model.save(self.path_model)

        loss, mse = model.evaluate(x_test, y_test, batch_size=1)
        print('loss: ', loss)
        print('mse: ', mse)

    def predict_test(self, x_test, y_test):
        model = load_model(self.path_model)
        y_pred = model.predict(x_test)
        print(f'###### test result ######')
        for i in range(100):
            print('predict: ', [int(_) for _ in y_pred[i]])
            print('truth: ', [int(_) for _ in y_test[i]])
            print('\n')


class LottoServices(object):
    def __init__(self):
        self.model1 = f'save/lotto_3t.h5'
        self.model2 = f'save/lotto_5t.h5'

    def process(self):
        pass

    def average(self):
        model = load_model(self.model1)



lotto_menus = ["Exit",  # 0
               "Modeling",  # 1
               "Service",  # 2
               ]

lotto_lambda = {
    "1": lambda t: LottoModel().process(int(input('time step: '))),
    "2": lambda t: t.process(),
    "3": lambda t: print(" ** No Function ** "),
    "4": lambda t: print(" ** No Function ** "),
    "5": lambda t: print(" ** No Function ** "),
    "6": lambda t: print(" ** No Function ** "),
    "7": lambda t: print(" ** No Function ** "),
    "8": lambda t: print(" ** No Function ** "),
    "9": lambda t: print(" ** No Function ** "),
}

if __name__ == '__main__':
    ls = LottoServices()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(lotto_menus)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("### Exit ###")
            break
        else:
            try:
                lotto_lambda[menu](ls)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")
