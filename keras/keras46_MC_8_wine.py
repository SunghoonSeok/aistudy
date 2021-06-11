import numpy as np
from sklearn.datasets import load_wine

dataset = load_wine()
# print(dataset.DESCR) 
# print(dataset.feature_names)

x = dataset.data
y = dataset.target
# print(x) 
# print(y) # 다중분류
# print(x.shape, y.shape) #(178, 13) (178, )
x_pred = x[-5:-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from tensorflow.keras.utils import to_categorical
# from keras.utils.up_utils import to_categorical
y = to_categorical(y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(200, (2,1), padding='same', input_shape=(13, 1, 1)))
model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(3, activation='softmax')) # softmax를 활용하면 node의 개수만큼 분리됨 분리된 값의 합은 1, 그중 가장 큰값을 선택

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # 다중분류일때 loss는 categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
modelpath= '../data/modelcheckpoint/k46_8_wine_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=400, validation_split=0.2, callbacks=[early_stopping,cp])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print("loss : ", loss)
print("acc : ", acc)


