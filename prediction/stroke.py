import numpy as np
import pandas as pd
from matplotlib import font_manager, rc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import joblib
font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5110 entries, 0 to 5109
Data columns (total 12 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   id                 5110 non-null   int64  
 1   gender             5110 non-null   object 
 2   age                5110 non-null   float64
 3   hypertension       5110 non-null   int64  
 4   heart_disease      5110 non-null   int64  
 5   ever_married       5110 non-null   object 
 6   work_type          5110 non-null   object 
 7   Residence_type     5110 non-null   object 
 8   avg_glucose_level  5110 non-null   float64
 9   bmi                4909 non-null   float64
 10  smoking_status     5110 non-null   object 
 11  stroke             5110 non-null   int64  
dtypes: float64(3), int64(4), object(5)
memory usage: 479.2+ KB
None
'''
stroke_meta = {
    'id':'아이디', 'gender':'성별', 'age':'나이',
    'hypertension':'고혈압',
    'heart_disease':'심장병',
    'ever_married':'기혼여부',
    'work_type':'직종',
    'Residence_type':'거주형태',
    'avg_glucose_level':'평균혈당',
    'bmi':'체질량지수',
    'smoking_status':'흡연여부',
    'stroke':'뇌졸중'
}

class StrokeService:
    def __init__(self):
        self.stroke = pd.read_csv('./data/healthcare-dataset-stroke-data.csv')
        self.model_path = f'./save/stroke_DecisionTree.pkl'
        self.my_stroke = None
        self.adult_stoke = None
        self.target = None
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.i = None
    '''
    1.스펙보기
    '''
    def spec(self):
        print(" --- 1.Shape ---")
        print(self.stroke.shape)
        print(" --- 2.Features ---")
        print(self.stroke.columns)
        print(" --- 3.Info ---")
        print(self.stroke.info())
        print(" --- 4.Case Top 3 ---")
        print(self.stroke.head(3))
        print(" --- 5.Case Bottom 3 ---")
        print(self.stroke.tail(3))
        print(" --- 6.Describe ---")
        print(self.stroke.describe())
    '''
    2.한글 메타데이터
    '''
    def rename_meta(self):
        self.my_stroke = self.stroke.rename(columns=stroke_meta)
        print(" --- 2.Features ---")
        print(self.my_stroke.columns)

    '''
    3.인터벌 = ['나이','평균혈당','체질량지수']
    '''
    def interval(self):
        t = self.my_stroke
        interval = ['나이', '평균혈당', '체질량지수']
        print(f'--- 구간변수 타입 --- \n {t[interval].dtypes}')
        print(f'--- 결측값 있는 변수 --- \n {t[interval].isna().any()[lambda x: x]}')
        print(f'체질량 결측비율: {t["체질량지수"].isnull().mean():.2f}')
        # 체질량 결측비율: 0.03 는 무시한다
        pd.options.display.float_format = '{:.2f}'.format
        print(f'--- 구간변수 기초 통계량 --- \n{t[interval].describe()}')
        criterion = t['나이'] > 18
        self.adult_stoke = t[criterion]
        print(f'--- 성인객체스펙 --- \n{self.adult_stoke.shape}')
        # 평균혈당 232.64이하와 체질량지수 60.3이하를 이상치로 규정하고 제거함
        t = self.adult_stoke
        c1 = t['평균혈당'] <= 232.64
        c2 = t['체질량지수'] <= 60.3
        self.adult_stoke = self.adult_stoke[c1 & c2]
        print(f'--- 이상치 제거한 성인객체스펙 ---\n{self.adult_stoke.shape}')

    def ratio(self): # 해당 컬럼 없음
        pass

    '''
    4.범주형 = ['성별', '심장병', '기혼여부', '직종', '거주형태','흡연여부', '고혈압']
    '''
    def norminal(self):
        t = self.adult_stoke
        category = ['성별', '심장병', '기혼여부', '직종', '거주형태', '흡연여부', '고혈압']
        print(f'범주형변수 데이터타입\n {t[category].dtypes}')
        print(f'범주형변수 결측값\n {t[category].isnull().sum()}')
        print(f'결측값 있는 변수\n {t[category].isna().any()[lambda x: x]}')# 결측값이 없음
        t['성별'] = OrdinalEncoder().fit_transform(t['성별'].values.reshape(-1,1))
        t['기혼여부'] = OrdinalEncoder().fit_transform(t['기혼여부'].values.reshape(-1, 1))
        t['직종'] = OrdinalEncoder().fit_transform(t['직종'].values.reshape(-1, 1))
        t['거주형태'] = OrdinalEncoder().fit_transform(t['거주형태'].values.reshape(-1, 1))
        t['흡연여부'] = OrdinalEncoder().fit_transform(t['흡연여부'].values.reshape(-1, 1))

        self.stroke = t
        self.spec()
        print(" ### 프리프로세스 완료 ### ")
        self.stroke.to_csv("./save/stroke.csv")

    def ordinal(self): # 해당 컬럼 없음
        pass

    '''
    데이터프레임을 데이터 파티션하기 전에 타깃변수와 입력변수를 
    target 과 data 에 분리하여 저장한다.
    '''
    def set_target(self):
        df = pd.read_csv('./save/stroke.csv')
        self.data = df.drop(['뇌졸중'], axis=1)
        self.data = self.data.drop(['Unnamed: 0'], axis=1)
        self.target = df['뇌졸중']
        print(f'--- data shape --- \n {self.data}')
        print(f'--- target shape --- \n {self.target}')

    def partition(self):
        data = self.data
        target = self.target
        undersample = RandomUnderSampler(sampling_strategy=0.333, random_state=2)
        data_under, target_under = undersample.fit_resample(data, target)
        print(target_under.value_counts(dropna=True))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_under, target_under,
                                                            test_size=0.5, random_state=42, stratify=target_under)
        print("X_train shape:", self.X_train.shape)
        print("X_test shape:", self.X_test.shape)
        print("y_train shape:", self.y_train.shape)
        print("y_test shape:", self.y_test.shape)

    def learning(self, flag):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        if flag == 'ccee':
            tree = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=5)
            tree.fit(X_train, y_train)
            joblib.dump(tree, self.model_path)
            print("Accuracy on training set: {:.5f}".format(tree.score(X_train, y_train)))
            print("Accuracy on test set: {:.5f}".format(tree.score(X_test, y_test)))

        elif flag == 'gini':
            tree = DecisionTreeClassifier(criterion="gini", random_state=0, max_depth=5)
            tree.fit(X_train, y_train)
            joblib.dump(tree, self.model_path)
            print("Accuracy on training set: {:.5f}".format(tree.score(X_train, y_train)))
            print("Accuracy on test set: {:.5f}".format(tree.score(X_test, y_test)))

        elif flag == 'grid':
            tree = DecisionTreeClassifier(criterion="gini", random_state=0, max_depth=5)
            tree.fit(X_train, y_train)
            params = {'criterion': ['gini', 'entropy'], 'max_depth': range(1,21)}
            # Grid Search Cross Validation
            # 5회의 교차검증을 2개의 기준마다 20개의 max_depth 값으로 대입 (5 * 2 * 20 = 총 500회) 학습(fit) 됨
            tree = GridSearchCV(
                tree,
                param_grid=params,
                scoring='accuracy',
                cv=5, # 교차 검증 파라미터 5회실시
                n_jobs=-1, # 멀티코어 CPU 모두 사용
                verbose=1 # 연산중간 메시지 출력
            )
            tree.fit(X_train, y_train)
            tree.fit(X_train, y_train)
            joblib.dump(tree, self.model_path)
            print(f"Accuracy on training set: {tree.score(X_train, y_train):.5f}")
            print(f"Accuracy on test set: {tree.score(X_test, y_test):.5f}")

            best_clf = tree.best_estimator_
            feature_names = list(self.data.columns)
            dft = pd.DataFrame(np.round(best_clf.feature_importances_, 4),
                               index=feature_names, columns=['Feature_importances'])
            dft1 = dft.sort_values(by="Feature_importances", ascending=False)
            print(dft1)
            ax = sns.barplot(y=dft1.index, x="Feature_importances", data=dft1)
            for p in ax.patches:
                ax.annotate("%.3f" % p.get_width(), (p.get_x() + p.get_width(), p.get_y() + 1),
                            xytext=(5, 10), textcoords='offset points')
            plt.show()


stroke_menu = ["Exit", #0
                "Spec",#1
                "Rename",#2
                "Inteval",#3 18세이상만 사용함
                "Norminal",#4
                "Set Target",#5
                "Partition",#6
                "Learning",#7
]
stroke_lambda = {
    "1" : lambda t: t.spec(),
    "2" : lambda t: t.rename_meta(),
    "3" : lambda t: t.interval(),
    "4" : lambda t: t.norminal(),
    "5" : lambda t: t.set_target(),
    "6" : lambda t: t.partition(),
    "7" : lambda t: t.learning(flag='grid'),
    "8" : lambda t: print(" ** No Function ** "),
    "9" : lambda t: print(" ** No Function ** "),
}


if __name__ == '__main__':
    ss = StrokeService()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(stroke_menu)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("### Exit ###")
            break
        else:
            try:
                t = None
                stroke_lambda[menu](ss)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")