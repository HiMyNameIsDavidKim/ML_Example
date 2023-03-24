from collections import Counter
import joblib
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.backend import expand_dims
from keras.optimizers import Adam
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.models import Sequential
from keras.layers import Dense, LSTM, ReLU, LayerNormalization, Dropout, GRU, Flatten, Multiply, Concatenate, RNN, \
    SimpleRNN, Conv1D, MaxPooling1D
from keras.saving.save import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
lotto_csv = './data/lotto_data.csv'
lotto_npy = './data/lotto_data.npy'
lotto_new_csv = './data/lotto_new_data.csv'
lotto_new_npy = './data/lotto_new_data.npy'
step = 2
epochs = 77
lr = 0.0001

compare = [7, 10, 22, 25, 34, 40]


def lr_schedule(epochs):
    lr = 0.0001
    if epochs > 20:
        lr *= 0.1
    if epochs > 40:
        lr *= 0.1
    if epochs > 60:
        lr *= 0.1
    return lr


class LottoModel(object):
    def __init__(self):
        self.df_lotto = pd.read_csv(lotto_csv, index_col=0, header=0, encoding='utf-8', sep=',')
        self.df = None
        self.path_model = f'./save/lotto_1hot/lotto_1hot_t{step}.h5'
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
        for i in range(6):
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
        df_numbers = self.df[:, 0:6]
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
        #
        # print('numbers')
        # print(f'X[0]: {str(self.onehot2num(self.x[0]))}')
        # print(f'Y[0]: {str(self.onehot2num(self.y[0]))}')

    def build_model(self):
        input1 = Input(shape=(step, 45))
        dense1 = Dense(128)(input1)
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
        preds = []
        truths = []
        for j in range(5):
            y_pred = model.predict(self.x_test)
            for i in range(len(y_pred)):
                pred = sorted(self.prob2num(y_pred[i]))
                truth = sorted(self.onehot2num(self.y_test[i][0]))
                preds.append(pred)
                truths.append(truth)
                inter = len(set(pred) & set(truth))
                inters.append(inter)
        for i in range(len(truths)//5):
            for j in range(5):
                print(f'predict{j+1}: {preds[i+j]}')
            print('\n')
            print(f'truth: {truths[i]}')
            for j in range(5):
                print(f'model got {inters[i+j]}ea number!')
            print('\n')
        self.inters = inters

    def inter2grade(self, inters):
        grade_index = ['꽝_0', '꽝_1', '꽝_2', '5동', '4등', '2/3등', '1등']
        grades = []
        for inter in inters:
            grade = grade_index[inter]
            grades.append(grade)
        return grades

    def predict_summary(self):
        grades = self.inter2grade(self.inters)
        print(f'1등 당첨 횟수: {Counter(grades)["1등"]}')
        print(f'2/3등 당첨 횟수: {Counter(grades)["2/3등"]}')
        print(f'4등 당첨 횟수: {Counter(grades)["4등"]}')
        print(f'5등 당첨 횟수: {Counter(grades)["5등"]}')
        print(f'2개 맞은 횟수: {Counter(grades)["꽝_2"]}')
        print(f'1개 맞은 횟수: {Counter(grades)["꽝_1"]}')
        print(f'0개 맞은 횟수: {Counter(grades)["꽝_0"]}')
        print(f'\n')


class LottoServices(object):
    def __init__(self):
        self.df_lotto = pd.read_csv(lotto_new_csv, index_col=0, header=0, encoding='utf-8', sep=',')
        self.list_step = [2, 2, 2, 2, 2]  # [1, 2, 8, 10, 12]
        self.model1 = f'./save/lotto_1hot/lotto_1hot_t{self.list_step[0]}.h5'
        self.model2 = f'./save/lotto_1hot/lotto_1hot_t{self.list_step[1]}.h5'
        self.model3 = f'./save/lotto_1hot/lotto_1hot_t{self.list_step[2]}.h5'
        self.model4 = f'./save/lotto_1hot/lotto_1hot_t{self.list_step[3]}.h5'
        self.model5 = f'./save/lotto_1hot/lotto_1hot_t{self.list_step[4]}.h5'
        self.list_model = [self.model1, self.model2, self.model3, self.model4, self.model5]
        self.df = None
        self.x = None
        self.grades = []
        self.ment = '★ ° . *　　　°　.　°☆ 　. * ☆ ¸. ★ ° :. ★　 * • ○ ° ★\n' \
                    '★ ° . *　　　°　.　°☆ 　. * ☆ ¸. ★ ° :. ★　 * • ○ ° ★\n' \
                    '★ ° . *　　　°　.　°안녕히 계세요 여러분~ :. ★　 * • ○ ° ★\n' \
                    '★ ° . *　전 이 세상의 모든 굴레와 속박을 벗어던지고 * • ○ ° ★\n' \
                    '★ ° . *　　　°　.제 행복을 찾아 떠납니다~~~! ★　 * • ○ ° ★\n' \
                    '★ ° . *　　　°　.　°☆ 　. * ☆ ¸. ★ ° :. ★　 * • ○ ° ★\n' \
                    '★ ° . *　　　°　.　°☆ 　. * ☆ ¸. ★ ° :. ★　 * • ○ ° ★'

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
        for i in range(6):
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
        return numbers

    def split(self, df, step):
        x = []
        tmp_x = df[-step:]
        x.append(tmp_x)
        return np.array(x)

    def dataset(self, step):
        df_numbers = self.df[-step:, 0:6]
        df_onehot = list(map(self.num2onehot, df_numbers))

        x = self.split(df_onehot, step)
        self.x = np.reshape(x, (x.shape[0], step, 45)).astype(float)

    def predict_this_week(self):
        results = f''
        for i in range(len(self.list_model)):
            [print(f'{self.ment}') for _ in range(10)]
            model = load_model(self.list_model[i])
            self.dataset(self.list_step[i])
            y = model.predict(self.x)
            pred = sorted(self.prob2num(y[0]))
            results += f'            {pred}\n'  # ({self.inter2grade(inter)})
            inter = len(set(pred) & set(compare))
            self.grades.append(self.inter2grade(inter))
        print('★ ° . *　　°　.　°☆ 　. * ☆ ¸. ★ ° :. ★　 * • ○ ° ★')
        print('★ ° . *　　°　.　°☆ 　. * ☆ ¸. ★ ° :. ★　 * • ○ ° ★\n')
        print(results)
        print('★ ° . *　　°　.　°☆ 　. * ☆ ¸. ★ ° :. ★　 * • ○ ° ★')
        print('★ ° . *　　°　.　°☆ 　. * ☆ ¸. ★ ° :. ★　 * • ○ ° ★\n')
        print(f'last week checker: {self.onehot2num(self.x[0][-1])}\n')

    def inter2grade(self, inter):
        grade_index = ['꽝_0', '꽝_1', '꽝_2', '5등', '4등', '2/3등', '1등']
        grade = grade_index[inter]
        return grade


def auto_run():
    list_step = [1, 2, 8, 10, 12]
    for s in list_step:
        global step
        step = s
        LottoModel().process_main()


def auto_test():
    cnt = 0
    ls = LottoServices()
    while True:
        ls.process()
        cnt += 1
        if ('1등' in ls.grades) or ('2/3등' in ls.grades) or ('4등' in ls.grades):
            print(f'###### {cnt}회 시행 결과 ######')
            print(f'1등: {ls.grades.count("1등")}회')
            print(f'2/3등: {ls.grades.count("2/3등")}회')
            print(f'4등: {ls.grades.count("4등")}회')
            print(f'5등: {ls.grades.count("5등")}회')
            print('\n')
            break


lotto_menus = ["Exit",  # 0
               "Modeling",  # 1
               "Test Model",  # 2
               "Predict This Week",  # 3
               "Auto Run",  # 4
               "Auto Test",  # 5
               ]

lotto_lambda = {
    "1": lambda t: LottoModel().process_main(),
    "2": lambda t: LottoModel().process_test(),
    "3": lambda t: t.process(),
    "4": lambda t: auto_run(),
    "5": lambda t: auto_test(),
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

'''
test_size=0.1
step1 = 0120
step2 = 1040
step4 = 0090
step6 = 0171
step8 = 0190
step10 = 0270
step12 = 0270
'''
'''
[2, 2, 2, 2, 2]
###### 72회 시행 결과 ###### 38%
1등: 0회
2/3등: 0회
4등: 1회
5등: 28회

[1, 1, 1, 1, 1]
###### 14회 시행 결과 ###### 36%
1등: 0회
2/3등: 0회
4등: 1회
5등: 5회

[8, 8, 8, 8, 8]
###### 99회 시행 결과 ###### 14%
1등: 0회
2/3등: 0회
4등: 1회
5등: 14회

[10, 10, 10, 10, 10]
###### 145회 시행 결과 ###### 20%
1등: 0회
2/3등: 0회
4등: 1회
5등: 29회

[12, 12, 12, 12, 12]
###### 93회 시행 결과 ###### 28%
1등: 0회
2/3등: 0회
4등: 1회
5등: 26회

[1, 1, 2, 2, 12]
###### 288회 시행 결과 ###### 21%
1등: 0회
2/3등: 0회
4등: 1회
5등: 62회

[2, 2, 2, 12, 12]
###### 28회 시행 결과 ###### 36%
1등: 0회
2/3등: 0회
4등: 1회
5등: 10회

[2, 2, 12, 12, 12]
###### 25회 시행 결과 ###### 36%
1등: 0회
2/3등: 0회
4등: 1회
5등: 9회

[2, 2, 1, 12, 12]
###### 39회 시행 결과 ###### 23%
1등: 0회
2/3등: 0회
4등: 1회
5등: 9회

[2, 2, 2, 1, 1]
###### 64회 시행 결과 ###### 20%
1등: 0회
2/3등: 0회
4등: 1회
5등: 13회

[2, 2, 8, 12, 12]
###### 40회 시행 결과 ###### 20%
1등: 0회
2/3등: 0회
4등: 1회
5등: 8회

[2, 2, 10, 12, 12]
###### 59회 시행 결과 ###### 20%
1등: 0회
2/3등: 0회
4등: 1회
5등: 12회
'''
'''
GRU
###### 72회 시행 결과 ###### 38%
1등: 0회
2/3등: 0회
4등: 1회
5등: 28회

LSTM
###### 85회 시행 결과 ###### 18%
1등: 0회
2/3등: 0회
4등: 1회
5등: 15회

RNN
100회 시행 당첨 실패

DNN
100회 시행 당첨 실패
'''