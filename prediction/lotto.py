from random import randint

import numpy as np
import pandas as pd
from keras import Input, Model
from keras.optimizers import Adam
from keras.optimizers.schedules.learning_rate_schedule import PiecewiseConstantDecay
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, LSTM, concatenate, Dropout, LayerNormalization, Softmax, Embedding, Flatten
from keras.saving.save import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

num_dict = {i: i-1 for i in range(1, 46)}
lotto_csv = './data/lotto_data.csv'
lotto_npy = './data/lotto_data.npy'
lotto_new_csv = './data/lotto_new_data.csv'
lotto_new_npy = './data/lotto_new_data.npy'
epochs = 60
lr = 0.0006
col = 1


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
        x1_train, x1_test, y1_train, y1_test = self.dataset(x1, y1, self.step)
        # self.build_model(x1_train, x1_test, y1_train, y1_test)
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
            tmp_x = df[i:x_end_number, col-1]
            tmp_y = df[x_end_number:y_end_number, col-1]
            x.append(tmp_x)
            y.append(tmp_y)
        return np.array(x), np.array(y)

    def dataset(self, x, y, step):
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3)
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train).astype(float)
        x_test_scaled = scaler.transform(x_test).astype(float)
        x_train = np.reshape(x_train_scaled, (x_train_scaled.shape[0], step, 1)).astype(float)
        x_test = np.reshape(x_test_scaled, (x_test_scaled.shape[0], step, 1)).astype(float)
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)
        return x_train, x_test, y_train, y_test

    def build_model(self, x_train, x_test, y_train, y_test):
        input1 = Input(shape=(self.step,))
        embedding = Embedding(input_dim=45, output_dim=45, input_length=self.step)(input1)
        flatten = Flatten()(embedding)
        dense1 = Dense(64)(flatten)
        dense1 = LayerNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)
        dense1 = Dense(32)(dense1)
        dense1 = LayerNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)
        output1 = Dense(1)(dense1)

        model = Model(inputs=input1, outputs=output1)
        model.compile(loss='mse', optimizer=Adam(learning_rate=lr), metrics=['mse'])

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
        list_good = []
        print(f'###### test result ######')
        for i in range(len(y_pred)):
            pred = round(y_pred[i][0])
            truth = int(y_test[i][0])
            print('predict: ', pred)
            print('truth: ', truth)
            print('\n')
            if pred == truth:
                list_good.append(pred)
        print(f'accuracy: {100*len(list_good)/len(y_pred):.2f}%')
        print('\n')


class LottoServices(object):
    def __init__(self):
        self.df = None
        self.x1 = None
        self.y1 = None
        self.model1 = f'./save/lotto/lotto_c{col}_t52.h5'
        self.model2 = f'./save/lotto/lotto_c{col}_t104.h5'
        self.model3 = f'./save/lotto/lotto_c{col}_t156.h5'
        self.model4 = f'./save/lotto/lotto_c{col}_t208.h5'
        self.model5 = f'./save/lotto/lotto_c{col}_t260.h5'
        self.model6 = f'./save/lotto/lotto_c{col}_t312.h5'
        self.model7 = f'./save/lotto/lotto_c{col}_t364.h5'
        self.list_model = [self.model1, self.model2, self.model3,
                           self.model4, self.model5, self.model6,
                           self.model7]
        self.list_step = [52, 104, 156, 208, 260, 312, 364]
        self.biases = [0, 0, 0, 0, 0, 0]

    def process_1(self):
        for i in range(1, 7):
            global col
            col = i
            self.show_result()

    def process_2(self):
        for i in range(1, 7):
            global col
            col = i
            self.make_csv()

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
            tmp_x = df[i:x_end_number, col-1]
            tmp_y = df[x_end_number:y_end_number, col-1]
            x.append(tmp_x)
            y.append(tmp_y)
        return np.array(x), np.array(y)

    def dataset(self, x, y, step):
        scaler = StandardScaler()
        scaler.fit(x)
        x_scaled = scaler.transform(x).astype(float)
        x = np.reshape(x_scaled, (x_scaled.shape[0], step, 1)).astype(float)
        y = y.astype(float)
        return x, y

    def show_result(self):
        self.np_read()
        results = []
        for num in range(len(self.list_model)):
            x1, y1 = self.split_xy(self.df, self.list_step[num])
            self.x1, self.y1 = self.dataset(x1, y1, self.list_step[num])
            model = load_model(self.list_model[num])
            y_pred = model.predict(self.x1)
            list_good = []
            for i in range(len(self.x1)):
                pred = round(y_pred[i][0])+self.biases[col-1]
                truth = int(self.y1[i][0])
                if pred == truth:
                    list_good.append(pred)
            results.append(f'time step {self.list_step[num]} accuracy: {100 * len(list_good) / len(y_pred):.2f}%')
        print(f'###### col: {col} ######')
        [print(result) for result in results]

    def make_csv(self):
        self.np_read()
        preds = pd.DataFrame(0, index=range(1057), columns=range(14))
        for num in range(len(self.list_model)):
            x1, y1 = self.split_xy(self.df, self.list_step[num])
            self.x1, self.y1 = self.dataset(x1, y1, self.list_step[num])
            model = load_model(self.list_model[num])
            y_pred = model.predict(self.x1)
            for i in range(len(self.x1)):
                pred = round(y_pred[-i][0])
                truth = int(self.y1[-i][0])
                preds[num][i] = pred
                preds[7+num][i] = truth
        preds.to_csv(f'./save/lotto_c{col}_preds.csv', index=False)

    def modify_csv(self):
        for c in range(1, 7):
            preds = pd.read_csv(f'./save/lotto_c{c}_preds.csv')
            preds.drop(0, inplace=True)
            preds.drop(columns=['8', '9', '10', '11', '12', '13'], inplace=True)
            preds = preds[preds['6'] != 0]
            preds.to_csv(f'./save/lotto_c{c}_preds.csv', index=False)

    def df2np(self):
        df = pd.read_csv(lotto_new_csv, index_col=0, header=0, encoding='utf-8', sep=',')
        df = df.drop(['bonus'], axis=1).dropna(axis=0)
        df = df.values
        np.save(lotto_new_npy, arr=df)

    def split_lotto(self, time_steps):
        df = np.load(lotto_new_npy, allow_pickle=True)
        x = []
        tmp_x = df[-1-time_steps:-1, col - 1]
        x.append(tmp_x)
        return np.array(x)

    def ds_lotto(self, x, step):
        scaler = StandardScaler()
        scaler.fit(x)
        x_scaled = scaler.transform(x).astype(float)
        x = np.reshape(x_scaled, (x_scaled.shape[0], step, 1)).astype(float)
        return x

    def verify_numbers(self, numbers):
        verified = []
        for number in numbers:
            if number < 1:
                number = 1
            elif number > 45:
                number = 45
            verified.append(number)
        return verified

    def lotto(self):
        lotto_1 = []
        '''
        각 컬럼별 베스트 모델의 조합
        [260, 364, 156, 156, 364, 52]
        '''
        model1 = f'./save/lotto/lotto_c1_t260.h5'
        model2 = f'./save/lotto/lotto_c2_t364.h5'
        model3 = f'./save/lotto/lotto_c3_t156.h5'
        model4 = f'./save/lotto/lotto_c4_t156.h5'
        model5 = f'./save/lotto/lotto_c5_t364.h5'
        model6 = f'./save/lotto/lotto_c6_t52.h5'
        list_model = [model1, model2, model3, model4, model5, model6]
        list_step = [260, 364, 156, 156, 364, 52]
        biases = [0, 4, 15, 25, 33, 38]

        self.df2np()
        for num in range(len(list_model)):
            x = self.split_lotto(list_step[num])
            x = self.ds_lotto(x, list_step[num])
            model = load_model(list_model[num])
            y = model.predict(x)
            y_b = round(y[0][0]) + biases[col-1]
            lotto_1.append(y_b)
        lotto_1 = self.verify_numbers(lotto_1)
        print(lotto_1)

def auto_run():
    for i in range(2, 7):
        global col, epochs, lr
        col = i
        epochs = col * 10
        lr = col/10000
        list_step = [52, 104, 156, 208, 260, 312, 364]
        for step in list_step:
            LottoModel().process(step)


lotto_menus = ["Exit",  # 0
               "Modeling",  # 1
               "Auto Run",  # 2
               "Show Summary",  # 3
               "Make CSV",  # 4
               "Modify CSV",  # 5
               "Ensemble",  # 6
               "Lotto",  # 7
               ]

lotto_lambda = {
    "1": lambda t: LottoModel().process(int(input('time step: '))),
    "2": lambda t: auto_run(),
    "3": lambda t: t.process_1(),
    "4": lambda t: t.process_2(),
    "5": lambda t: t.modify_csv(),
    "6": lambda t: print(" ** No Function ** "),
    "7": lambda t: t.lotto(),
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
