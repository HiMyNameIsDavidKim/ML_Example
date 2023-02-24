import pickle

import numpy as np
import pandas as pd
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, LSTM, concatenate
from keras.saving.save import load_model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class StockLSTM(object):
    def __init__(self, fit_refresh):
        global kospi_csv, sam_csv, kospi_npy, sam_npy, lstm_model
        kospi_csv = './data/kospi_data.csv'
        sam_csv = './data/samsung_data.csv'
        kospi_npy = './data/kospi_data.npy'
        sam_npy = './data/samsung_data.npy'
        lstm_model = './save/stock_predict_LSTM.h5'
        self.fit_refresh = fit_refresh
        self.df_kospi = pd.read_csv(kospi_csv, index_col=0, header=0, encoding='utf-8', sep=',')
        self.df_sam = pd.read_csv(sam_csv, index_col=0, header=0, encoding='utf-8', sep=',')
        self.kospi200 = None
        self.samsung = None

    def process(self):
        self.df_modify(self.df_kospi, self.df_sam)
        self.np_read()
        self.process_lstm(self.fit_refresh)
        self.pred_test(lstm_model, x_test_lstm, y_test_lstm)

    def df_modify(self, df1, df2):
        # print(df1, df1.shape)
        # print(df2, df2.shape)

        df1 = df1.drop(['변동 %'], axis=1).dropna(axis=0)
        df2 = df2.drop(['변동 %'], axis=1).dropna(axis=0)
        for i in range(len(df1.index)):
            if type(df2.iloc[i, 4]) == float:
                pass
            if 'K' in df1.iloc[i, 4]:
                df1.iloc[i, 4] = int(float(df1.iloc[i, 4].replace('K', '')) * 1000)
            elif 'M' in df1.iloc[i, 4]:
                df1.iloc[i, 4] = int(float(df1.iloc[i, 4].replace('M', '')) * 1000 * 1000)
        for i in range(len(df2.index)):
            if type(df2.iloc[i, 4]) == float:
                pass
            elif 'K' in df2.iloc[i, 4]:
                df2.iloc[i, 4] = int(float(df2.iloc[i, 4].replace('K', '')) * 1000)
            elif 'M' in df2.iloc[i, 4]:
                df2.iloc[i, 4] = int(float(df2.iloc[i, 4].replace('M', '')) * 1000 * 1000)
        for i in range(len(df2.index)):
            for j in range(len(df2.iloc[i]) - 1):
                df2.iloc[i, j] = int(df2.iloc[i, j].replace(',', ''))
        df1 = df1.sort_values(['날짜'], ascending=[True])
        df2 = df2.sort_values(['날짜'], ascending=[True])

        df1 = df1.values
        df2 = df2.values
        np.save(kospi_npy, arr=df1)
        np.save(sam_npy, arr=df2)
        np_df1 = np.array(df1)
        np_df2 = np.array(df2)
        return np_df1, np_df2

    def np_read(self):
        kospi200 = np.load(kospi_npy, allow_pickle=True)
        samsung = np.load(sam_npy, allow_pickle=True)
        # print(kospi200, kospi200.shape)
        # print(samsung, samsung.shape)
        self.kospi200 = kospi200
        self.samsung = samsung

    def split_xy(self, dataset, time_steps, y_column):
        x, y = list(), list()
        for i in range(len(dataset)):
            x_end_number = i + time_steps
            y_end_number = x_end_number + y_column

            if y_end_number > len(dataset):
                break
            tmp_x = dataset[i:x_end_number, :]
            tmp_y = dataset[x_end_number:y_end_number, 3]
            x.append(tmp_x)
            y.append(tmp_y)
        return np.array(x), np.array(y)

    def process_lstm(self, fit_refresh):
        x, y = self.split_xy(self.samsung, 4, 1)
        x_train_scaled, x_test_scaled, y_train, y_test = self.dataset_lstm(x, y)
        if fit_refresh:
            self.modeling_lstm(lstm_model, x_train_scaled, x_test_scaled, y_train, y_test)
        global x_test_lstm, y_test_lstm
        x_test_lstm, y_test_lstm = x_test_scaled, y_test

    def dataset_lstm(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.3)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train).astype(float)
        x_test_scaled = scaler.transform(x_test).astype(float)
        x_train_scaled = np.reshape(x_train_scaled, (x_train_scaled.shape[0], 4, 5)).astype(float)
        x_test_scaled = np.reshape(x_test_scaled, (x_test_scaled.shape[0], 4, 5)).astype(float)
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)
        return x_train_scaled, x_test_scaled, y_train, y_test

    def modeling_lstm(self, name, x_train_scaled, x_test_scaled, y_train, y_test):
        model = Sequential()
        model.add(LSTM(64, input_shape=(4, 5)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])

        early_stopping = EarlyStopping(patience=20)
        history = model.fit(x_train_scaled, y_train, validation_split=0.2, verbose=1,
                            batch_size=1, epochs=100, callbacks=[early_stopping])
        model.save(name)
        df = pd.DataFrame(history.history)
        df.to_csv('./save/lstm_loss.csv')

        loss, mse = model.evaluate(x_test_scaled, y_test, batch_size=1)
        print('loss: ', loss)
        print('mse: ', mse)

    def pred_test(self, name, x_test_scaled, y_test):
        model = load_model(name)
        y_pred = model.predict(x_test_scaled)
        print(f'-----predict test by {name[21:-3]}-----')
        for i in range(5):
            print('close: ', int(y_test[i]), ' / ', 'predict: ', int(y_pred[i]))
        return str(int(y_test[0]))

    def loss_plot(self):
        df = pd.read_csv('./save/lstm_loss.csv')
        plt.plot(df['loss'], label='loss')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    StockLSTM(fit_refresh=False).process()
    # StockLSTM(fit_refresh=False).loss_plot()
