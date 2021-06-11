# 네이밍 룰
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
# from tensorflow.keras import models
# from tensorflow import keras
from tensorflow.keras.layers import Dense

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8])
y_train = np.array([1,2,3,4,5,6,7,8])
x_test = np.array([9,10,11])
y_test = np.array([9,10,11])

#2. 모델구성
model = Sequential()
# model = models.Sequential()
# model = keras.models.Sequential()
model.add(Dense(30, input_dim=1, activation='relu'))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)

result = model.predict([12])
print("result : ", result)

