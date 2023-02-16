import keras.datasets.fashion_mnist
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import os


class FashionModel(object):
    def __init__(self):
        global model_path
        model_path = f'./save/fashion_DNN.h5'

    def process(self):
        self.dataset()
        self.show()
        self.create_model()

    def dataset(self):
        (self.train_images, self.train_labels), \
            (self.test_images, self.test_labels) = keras.datasets.fashion_mnist.load_data()

    def show(self):
        plt.figure()
        plt.imshow(self.train_images[10])
        plt.colorbar()
        plt.grid(False)
        plt.show()

    def create_model(self):
        model = Sequential([
            keras.layers.Flatten(input_shape=(28,28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(self.train_images, self.train_labels, epochs=5)
        test_loss, test_acc = model.evaluate(self.test_images, self.test_labels)
        print(f'Test Accuracy is {test_acc}')
        model.save(model_path)


class FashionService(object):
    def __init__(self):
        global class_names, model_path
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        model_path = f'./save/fashion_DNN.h5'
        self.i = None

    def process(self):
        self.dataset()
        self.idx()
        self.eval_model()
        self.show()

    def dataset(self):
        (self.train_images, self.train_labels), \
            (self.test_images, self.test_labels) = keras.datasets.fashion_mnist.load_data()

    def idx(self):
        print(f'item No range : 0~{len(self.test_images)}')
        self.i = int(input('Input item No : '))

    def eval_model(self):
        i = self.i
        model = keras.models.load_model(model_path)
        predictions = model.predict(self.test_images)
        predictions_array, true_label, img = predictions[i], self.test_labels[i], self.test_images[i]

        result = np.argmax(predictions_array)
        print(f"###### 예측한 답 : {result} ######")
        print(f'###### {class_names[result]} ######')

    def show(self):
        i = self.i
        plt.figure()
        plt.imshow(self.test_images[i])
        plt.colorbar()
        plt.grid(False)
        plt.show()


fashion_menus = ["Exit",  # 0
                 "Modeling",  # 1
                 "Service",  # 2
]
fashion_lambda = {
    "1": lambda t: FashionModel().process(),
    "2": lambda t: FashionService().process(),
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
        [print(f"{i}. {j}") for i, j in enumerate(fashion_menus)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("### Exit ###")
            break
        else:
            try:
                t = None
                fashion_lambda[menu](t)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")