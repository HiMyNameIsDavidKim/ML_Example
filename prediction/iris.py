import pandas as pd
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np


class IrisModel(object):
    def __init__(self):
        global model_path
        model_path = f'./save/iris_DNN.h5'
        self.iris = datasets.load_iris()
        self.my_iris = None
        self._X = self.iris.data
        self._Y = self.iris.target

    def process(self):
        self.spec()
        self.create_model()

    def spec(self):
        print(f'{self.iris.feature_names}')

    def create_model(self):
        X = self._X
        Y = self._Y
        enc = OneHotEncoder()
        Y_1hot = enc.fit_transform(Y.reshape(-1, 1)).toarray()

        model = Sequential()
        model.add(Dense(4, input_dim=4, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(X, Y_1hot, epochs=300, batch_size=10)
        print('Model Training is completed.')

        model.save(model_path)


class IrisService(object):
    def __init__(self):
        global model, graph, target_names, class_names
        model_path = f'./save/iris_DNN.h5'
        model = keras.models.load_model(model_path)
        # graph = tf.get_default_graph()
        target_names = datasets.load_iris().target_names
        class_names = ['setosa, 부채붓꽃', 'versicolor, 버시칼라', 'virginica, 버지니카']
        self.features = None

    def process(self):
        self.input_features()
        self.service_model()

    def input_features(self):
        features = list(map(int, input('Please input 4ea features : ').split()))
        self.features = np.reshape(features, (1, 4))

    def service_model(self):
        Y_prob = model.predict(self.features, verbose=0)
        predicted = Y_prob.argmax(axis=-1)
        result = predicted[0]
        print(f'###### species: {class_names[result]} ######')


iris_menus = ["Exit",  # 0
              "Modeling",  # 1
              "Service",  # 2
              ]
iris_lambda = {
    "1": lambda t: IrisModel().process(),
    "2": lambda t: IrisService().process(),
    "3": lambda t: print(" ** No Function ** "),
    "4": lambda t: print(" ** No Function ** "),
    "5": lambda t: print(" ** No Function ** "),
    "6": lambda t: print(" ** No Function ** "),
    "7": lambda t: print(" ** No Function ** "),
    "8": lambda t: print(" ** No Function ** "),
    "9": lambda t: print(" ** No Function ** "),
}

if __name__ == '__main__':
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(iris_menus)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("### Exit ###")
            break
        else:
            try:
                t = None
                iris_lambda[menu](t)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")
