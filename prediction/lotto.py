import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense


class SeqPredictor:
    def __init__(self, seq_length, lstm_units, dense_units, learning_rate, epochs, batch_size, verbose):
        self.seq_length = seq_length
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(self.lstm_units, input_shape=(self.seq_length, 1)))
        model.add(Dense(self.dense_units))
        model.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def fit(self, X_train, y_train):
        X_train = tf.keras.utils.normalize(X_train)
        X_train = tf.reshape(X_train, (len(X_train), self.seq_length, 1))
        y_train = tf.reshape(y_train, (len(y_train), 1))
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

    def predict(self, next_seq):
        next_seq = tf.keras.utils.normalize(next_seq)
        next_seq = tf.reshape(next_seq, (1, self.seq_length, 1))
        pred = self.model.predict(next_seq)
        return pred[0][0]


if __name__ == '__main__':
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    X = []
    y = []
    for i in range(2, len(data)):
        X.append(data[i - 2:i])
        y.append(data[i])

    X_train = tf.constant(X)
    y_train = tf.constant(y)

    seq_predictor = SeqPredictor(seq_length=3,
                                 lstm_units=64,
                                 dense_units=1,
                                 learning_rate=0.01,
                                 epochs=1000,
                                 batch_size=1,
                                 verbose=0)
    seq_predictor.fit(X_train, y_train)

    next_seq = [[10, 11, 12]]
    pred = seq_predictor.predict(next_seq)
    print('다음 시퀸스 예측 결과: ', pred)
