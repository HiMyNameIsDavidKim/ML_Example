from collections import Counter
import joblib
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.optimizers import Adam
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.models import Sequential
from keras.layers import Dense, LSTM, ReLU, LayerNormalization, Dropout, GRU
from keras.saving.save import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


lotto_csv = './data/lotto_data.csv'
lotto_npy = './data/lotto_data.npy'
lotto_new_csv = './data/lotto_new_data.csv'
lotto_new_npy = './data/lotto_new_data.npy'
col = 2
step = 1
epochs = 20
lr = 0.0001


'''
c1
step1 = 19
step4 = 22  3rd
step26 = 26  1st
step52 = 20
step104 = 26  2nd
c2
step1 = 
step4 = 
step26 = 
step52 = 
step104 = 
c3
step1 = 
step4 = 
step26 = 
step52 = 
step104 = 
c4
step1 = 
step4 = 
step26 = 
step52 = 
step104 = 
c5
step1 = 
step4 = 
step26 = 
step52 = 
step104 = 
c6
step1 = 
step4 = 
step26 = 
step52 = 
step104 = 
'''

def lr_schedule(epochs):
    lr = 0.00002
    if epochs >= 10:
        lr *= 0.1
    # if epochs >= 20:
    #     lr *= 0.1
    return lr


class LottoModel(object):
    def __init__(self):
        self.df_lotto = pd.read_csv(lotto_csv, index_col=0, header=0, encoding='utf-8', sep=',')
        self.df = None
        self.path_model = f'./save/lotto_1by1/lotto_1by1_c{col}_t{step}.h5'
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.inters = None

    def process_main(self):
        self.df2np()
        self.np_read()
        self.dataset()
        self.build_model()

    def process_test(self):
        self.df2np()
        self.np_read()
        self.dataset()
        self.predict_test()
        self.predict_summary()

    def df2np(self):
        df = self.df_lotto
        df = df.drop(['bonus'], axis=1).dropna(axis=0)
        df = df.values
        np.save(lotto_npy, arr=df)

    def np_read(self):
        df = np.load(lotto_npy, allow_pickle=True)
        self.df = df

    def num2onehot(self, numbers):
        onehot = np.zeros(45)
        for i in range(1):
            onehot[int(numbers[i])-1] = 1
        return onehot

    def onehot2num(self, onehot):
        numbers = []
        for i in range(len(onehot)):
            if onehot[i] == 1.0:
                numbers.append(i+1)
        return numbers

    def prob2num(self, prob):
        temp = []
        numbers = []
        for n in range(45):
            cnt = int(prob[n] * 100 + 1)
            ball = np.full(cnt, n + 1)
            temp += list(ball)
        while True:
            if len(numbers) == 6:
                break
            ball_index = np.random.randint(len(temp), size=1)[0]
            ball = temp[ball_index]
            if ball not in numbers:
                numbers.append(ball)
        numbers = [sorted(numbers)[col - 1]]
        return numbers

    def split(self, df, step):
        x, y = [], []
        for i in range(len(df)):
            x_end_number = i + step
            y_end_number = x_end_number + 1
            if y_end_number > len(df):
                break
            tmp_x = df[i:x_end_number]
            tmp_y = df[x_end_number:y_end_number]
            x.append(tmp_x)
            y.append(tmp_y)
        return np.array(x), np.array(y)

    def dataset(self):
        df_numbers = self.df[:, col-1:col]
        df_onehot = list(map(self.num2onehot, df_numbers))

        x, y = self.split(df_onehot, step)

        x_train, x_test, \
            y_train, y_test = train_test_split(x, y,
                                               random_state=42,
                                               test_size=0.2)
        self.x_train = np.reshape(x_train, (x_train.shape[0], step, 45)).astype(float)
        self.x_test = np.reshape(x_test, (x_test.shape[0], step, 45)).astype(float)
        self.y_train = np.reshape(y_train, (y_train.shape[0], 1, 45)).astype(float)
        self.y_test = np.reshape(y_test, (y_test.shape[0], 1, 45)).astype(float)

        # print('onehot')
        # print(f'X[0]: {str(self.x[0])}')
        # print(f'Y[0]: {str(self.y[0])}')
        # print('numbers')
        # print(f'X[0]: {str(self.onehot2num(self.x[0]))}')
        # print(f'Y[0]: {str(self.onehot2num(self.y[0]))}')

    def build_model(self):
        input1 = Input(shape=(step, 45))
        dense1 = GRU(128)(input1)
        dense1 = LayerNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)
        dense1 = Dense(64)(dense1)
        dense1 = LayerNormalization()(dense1)
        dense1 = Dropout(0.3)(dense1)
        dense1 = Dense(1000)(dense1)
        output1 = Dense(45, activation='relu')(dense1)

        model = Model(inputs=input1, outputs=output1)
        model.compile(loss='mse', optimizer=Adam(learning_rate=lr), metrics=['mse'])

        lr_scheduler = LearningRateScheduler(lr_schedule)
        early_stopping = EarlyStopping(patience=20)
        model.fit(self.x_train, self.y_train,
                  validation_split=0.2, verbose=1, batch_size=1, epochs=epochs,
                  callbacks=[lr_scheduler])
        model.save(self.path_model)

        loss, mse = model.evaluate(self.x_test, self.y_test, batch_size=1)
        print('loss: ', loss)
        print('mse: ', mse)

    def predict_test(self):
        model = load_model(self.path_model)
        inters = []
        y_pred = model.predict(self.x_test)
        for i in range(len(y_pred)):
            pred = sorted(self.prob2num(y_pred[i]))
            truth = sorted(self.onehot2num(self.y_test[i][0]))
            print(f'predict: {pred}')
            print(f'truth: {truth}')
            inter = len(set(pred) & set(truth))
            print(f'model got {inter}ea number!')
            print('\n')
            inters.append(inter)
        self.inters = inters

    def inter2grade(self, inters):
        grade_index = ['꽝', '정답']
        grades = []
        for inter in inters:
            grade = grade_index[inter]
            grades.append(grade)
        return grades

    def predict_summary(self):
        grades = self.inter2grade(self.inters)
        print(f'정답 횟수: {Counter(grades)["정답"]}')
        print(f'꽝 횟수: {Counter(grades)["꽝"]}')
        print(f'\n')


class LottoServices(object):
    def __init__(self):
        self.df_lotto = pd.read_csv(lotto_new_csv, index_col=0, header=0, encoding='utf-8', sep=',')
        self.model1 = f'./save/lotto_1by1/lotto_1by1_c1_t26.h5'
        self.model2 = f'./save/lotto_1by1/lotto_1by1_c1_t104.h5'
        self.model3 = f'./save/lotto_1by1/lotto_1by1_c1_t4.h5'
        self.list_model = [self.model1, self.model2, self.model3,
                           ]
        self.list_step = [26, 104, 4,
                          ]
        self.df = None
        self.x = None

    def process(self):
        self.df2np()
        self.np_read()
        self.predict_this_week()

    def df2np(self):
        df = self.df_lotto
        df = df.drop(['bonus'], axis=1).dropna(axis=0)
        df = df.values
        np.save(lotto_new_npy, arr=df)

    def np_read(self):
        df = np.load(lotto_new_npy, allow_pickle=True)
        self.df = df

    def num2onehot(self, numbers):
        onehot = np.zeros(45)
        for i in range(1):
            onehot[int(numbers[i])-1] = 1
        return onehot

    def onehot2num(self, onehot):
        numbers = []
        for i in range(len(onehot)):
            if onehot[i] == 1.0:
                numbers.append(i+1)
        return numbers

    def prob2num(self, prob):
        temp = []
        numbers = []
        for n in range(45):
            cnt = int(prob[n] * 100 + 1)
            ball = np.full(cnt, n + 1)
            temp += list(ball)
        while True:
            if len(numbers) == 6:
                break
            ball_index = np.random.randint(len(temp), size=1)[0]
            ball = temp[ball_index]
            if ball not in numbers:
                numbers.append(ball)
        numbers = [sorted(numbers)[col - 1]]
        return numbers

    def split(self, df, step):
        x = []
        tmp_x = df[-step:]
        x.append(tmp_x)
        return np.array(x)

    def dataset(self, step):
        df_numbers = self.df[-step:, col-1:col]
        df_onehot = list(map(self.num2onehot, df_numbers))

        x = self.split(df_onehot, step)
        self.x = np.reshape(x, (x.shape[0], step, 45)).astype(float)

    def predict_this_week(self):
        results = f''
        for i in range(len(self.list_model)):
            model = load_model(self.list_model[i])
            self.dataset(self.list_step[i])
            y = model.predict(self.x)
            pred = self.prob2num(y[0])
            results += f'good luck bro! {pred}\n'
        preds = []
        for i in range(len(self.list_model)):
            model = load_model(self.list_model[i])
            self.dataset(self.list_step[i])
            y = model.predict(self.x)
            pred = self.prob2num(y[0])
            preds.append(pred)
        results += f'good luck bro! {(preds[0][0] + preds[1][0])//2}\n'
        results += f'good luck bro! {(preds[0][0] + preds[1][0] + preds[2][0])//3}\n'
        print(results)


def auto_run():
    for c in [2, 3, 4, 5, 6]:
        global col
        col = c
        for s in [1, 4, 26, 52, 104]:
            global step
            step = s
            LottoModel().process_main()


lotto_menus = ["Exit",  # 0
               "Modeling",  # 1
               "Test Model",  # 2
               "Predict This Week",  # 3
               "Auto Run",  # 4
               ]

lotto_lambda = {
    "1": lambda t: LottoModel().process_main(),
    "2": lambda t: LottoModel().process_test(),
    "3": lambda t: t.process(),
    "4": lambda t: auto_run(),
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
