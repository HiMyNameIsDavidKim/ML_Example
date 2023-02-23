import os
import keras.datasets.mnist
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import keras.datasets.mnist


class NumberModel(object):
    def __init__(self):
        global model_path
        model_path = f'./save/number_DNN.h5'

    def process(self):
        self.dataset()
        self.create_model()

    def dataset(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

    def create_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(self.x_train, self.y_train, epochs=5)

        test_loss, test_acc = model.evaluate(self.x_test, self.y_test)
        print('테스트 정확도:', f'{test_acc * 100:.2f}%')

        model.save(model_path)


class NumberService(object):
    def __init__(self):
        global class_names, model_path
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        model_path = f'./save/number_DNN.h5'
        self.i = None

    def process(self):
        self.dataset()
        self.idx()
        self.service_model()
        self.show()

    def dataset(self):
        (self.train_images, self.train_labels), \
            (self.test_images, self.test_labels) = keras.datasets.mnist.load_data()

    def idx(self):
        print(f'test number range : 0~{len(self.test_images)}')
        self.i = int(input('input test number : '))

    def service_model(self):
        i = self.i
        model = keras.models.load_model(model_path)
        predictions = model.predict(self.test_images)
        predictions_array, true_label, img = predictions[i], self.test_labels[i], self.test_images[i]

        result = np.argmax(predictions_array)
        print(f"예측한 답 : {str(result)}")

    def show(self):
        i = self.i
        plt.figure()
        plt.imshow(self.test_images[i])
        plt.colorbar()
        plt.grid(False)
        plt.show()


number_menus = ["Exit", # 0
               "Modeling", # 1
               "Service", # 2
]
number_lambda = {
    "1": lambda t: NumberModel().process(),
    "2": lambda t: NumberService().process(),
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
        [print(f"{i}. {j}") for i, j in enumerate(number_menus)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("### Exit ###")
            break
        else:
            try:
                t = None
                number_lambda[menu](t)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")