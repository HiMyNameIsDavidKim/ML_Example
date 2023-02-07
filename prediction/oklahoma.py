import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class Oklahoma(object):
    def __init__(self):
        self.df = pd.read_csv('./data/house-2017.csv')

    def process(self):
        self.data_pre()

    def data_pre(self):
        pass




if __name__ == '__main__':
    Oklahoma().process()