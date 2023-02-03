from prediction.stock_dnn import StockDNN
from prediction.stock_dnn_ensemble import StockDNNEnsemble
from prediction.stock_lstm import StockLSTM
from prediction.stock_lstm_ensemble import StockLSTMEnsemble

stock_menus = ["Exit", # 0
               "DNN", # 1
               "LSTM", # 2
               "DNN_Ensemble",  # 3
               "LSTM_Ensemble", # 4
]
stock_lambda = {
    "1": lambda t: StockDNN(fit_refresh=False).process(),
    "2": lambda t: StockLSTM(fit_refresh=False).process(),
    "3": lambda t: StockDNNEnsemble(fit_refresh=False).process(),
    "4": lambda t: StockLSTMEnsemble(fit_refresh=False).process(),
    "5": lambda t: print(" ** No Function ** "),
    "6": lambda t: print(" ** No Function ** "),
    "7": lambda t: print(" ** No Function ** "),
    "8": lambda t: print(" ** No Function ** "),
    "9": lambda t: print(" ** No Function ** "),
}


if __name__ == '__main__':
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(stock_menus)]
        menu = input('Choose menu : ')
        if menu == '0':
            print("### Exit ###")
            break
        else:
            try:
                t = None
                stock_lambda[menu](t)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message.')
                else:
                    print("Didn't catch error message.")