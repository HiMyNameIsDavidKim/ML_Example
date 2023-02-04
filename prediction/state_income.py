import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split


class StateIncome(object):
    def __init__(self):
        self.df_adult = pd.read_csv('./data/adult.csv')
        self.df_train = None
        self.df_test = None
        self.model = None
        self.train_x = None
        self.train_y = None

    def process(self):
        self.pre_df()
        self.data_split()
        self.modeling()
        # self.plot_structure()
        self.predict()

    def pre_df(self):
        df = self.df_adult
        df['income'] = np.where(df['income'] == '>50K', 'high', 'low')
        df['income'].value_counts(normalize=True)
        df = df.drop(columns='fnlwgt')

        target = df['income']
        df = df.drop(columns='income')
        df = pd.get_dummies(df)
        df['income'] = target

        print('###### data type ######')
        print(df.dtypes)
        self.df_adult = df

    def data_split(self):
        df = self.df_adult
        self.df_train, self.df_test = train_test_split(df,
                                                       test_size=0.3,
                                                       stratify=df['income'],
                                                       random_state=42)
        print('###### data split ######')
        print(f'train set : {self.df_train.shape}')
        print(f'test set : {self.df_test.shape}')

    def modeling(self):
        clf = tree.DecisionTreeClassifier(random_state=42, max_depth=3)
        self.train_x = self.df_train.drop(columns='income')
        self.train_y = self.df_train['income']

        self.model = clf.fit(X=self.train_x, y=self.train_y)
        print('###### completed ######')

    def plot_structure(self):
        plt.rcParams.update({'figure.dpi': '100',  # 그래프 크기 설정
                             'figure.figsize': [12, 8]})  # 해상도 설정
        tree.plot_tree(self.model,
                       feature_names=self.train_x.columns,  # 예측 변수명
                       class_names=['high', 'low'],  # 타겟 변수 클래스, 알파벳순
                       proportion=True,  # 비율 표기
                       filled=True,  # 색칠
                       rounded=True,  # 둥근 테두리
                       impurity=False,  # 불순도 표시
                       label='root',  # label 표시 위치
                       fontsize=10)  # 글자 크기
        plt.show()

    def predict(self):
        test_x = self.df_test.drop(columns='income')
        test_y = self.df_test['income']

        accuracy = metrics.accuracy_score(y_true=test_y, y_pred=self.model.predict(test_x))
        print(f'predict accuracy : {accuracy:.2f}')


state_income_menu = ["Exit",  # 0
                     "Preprocess",  # 1
                     "Split Data",  # 2
                     "Modeling",  # 3
                     "Plot Structure",  # 4
                     "Predict",  # 5
                     ]
state_income_lambda = {
    "1": lambda t: t.pre_df(),
    "2": lambda t: t.data_split(),
    "3": lambda t: t.modeling(),
    "4": lambda t: t.plot_structure(),
    "5": lambda t: t.predict(),
    "6": lambda t: print(" ** No Function ** "),
    "7": lambda t: print(" ** No Function ** "),
    "8": lambda t: print(" ** No Function ** "),
    "9": lambda t: print(" ** No Function ** "),
}

if __name__ == '__main__':
    si = StateIncome()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(state_income_menu)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("### Exit ###")
            break
        else:
            try:
                state_income_lambda[menu](si)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")
