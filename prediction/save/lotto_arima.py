import numpy as np
import statsmodels.api as sm

# 입력 데이터 생성
X = np.random.rand(1059, 6)

# ARIMA 모델 생성
model = sm.tsa.ARIMA(X, order=(1, 0, 0))

# 모델 학습
results = model.fit()

# 다음 값을 예측
pred = results.forecast()[0]

print(pred)

'''
ARIMA 모델은 AR(p), MA(q), ARMA(p,q), ARIMA(p,d,q) 모델들의 일반화된 형태입니다. 
ARIMA 모델은 AR 모델과 MA 모델이 결합된 모델로서, 
이전 관측값들과 이전 예측 오차들 모두를 이용하여 현재 값을 예측합니다. 
ARIMA 모델에서 p, d, q값을 적절히 조절하면 시계열 데이터의 특성을 잘 반영하는 모델을 만들 수 있습니다.
'''