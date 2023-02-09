import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class Oklahoma(object):
    def __init__(self):
        self.df = pd.read_csv('./data/comb31-IQR30.csv')
        self.comb1 = None

    def process(self):
        pass




if __name__ == '__main__':
    Oklahoma().process()
