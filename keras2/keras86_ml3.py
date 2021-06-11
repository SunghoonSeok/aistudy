# # 실습
# # 1. 상단모델에 그리드 서치 또는 랜덤서치로 튜닝한 모델 구성
# # 최적의 R2값과 피처 임포턴스 구할 것

# # 2. 위 쓰레드 값으로 selectfrommodel 을 구해서
# # 최적의 피처 갯수를 구할것

# # 3. 위 피처 갯수로 데이터(피처)를 수정(삭제)해서
# # 그리드 서치 또는 랜덤서치 적용하여
# # 최적의 R2 구할것

# # 1번값과 비교

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score


import pandas as pd

df = pd.read_csv('c:/data/csv/winequality-white.csv', delimiter=';')
print(df.corr)
print(df.head())
print(df.index)

print(df.shape) # 4898,12

x = df.iloc[:,:-1]
y = df.iloc[:,-1]
print(x.shape, y.shape)

print(y.unique()) # 7
print(y.value_counts())

found1 = df[df['quality']==3].index
found2 = df[df['quality']==9].index

df = df.drop(found1)
df = df.drop(found2)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

print(y.unique())
print(y.value_counts())
print(x.shape, y.shape)


x = x.values
y = y.values

# from sklearn.preprocessing import OneHotEncoder

# y = y.reshape(-1,1)

# ohencoder = OneHotEncoder()
# ohencoder.fit(y)
# y = ohencoder.transform(y).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=42)


from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, PowerTransformer
# scaler = QuantileTransformer(n_quantiles=100)
# scaler = PowerTransformer()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


parameters = [
    {'n_estimators':[1000,1200,800], 'learning_rate': [0.1, 0.05, 0.01], 'max_depth':[3,4,5]},
    {'n_estimators':[900,400,1100], 'learning_rate': [0.1, 0.01, 0.001], 'max_depth':[4,5,6], 'colsample_bytree':[0.8, 0.9, 1]},
    {'n_estimators':[900,1100,500], 'learning_rate': [0.1, 0.01], 'max_depth':[4,5,6], 'colsample_bytree':[0.7, 0.9, 1], 'colsample_bylebel':[0.7,0.9,1]}
]

model = RandomizedSearchCV(XGBRegressor(tree_method='gpu_hist',predictor='gpu_predictor',gpu_id=0), parameters, cv=5)
model.fit(x_train,y_train)

print("모델 스코어는 : ", model.score(x_test,y_test))
print("최적의 모델은? : ", model.best_estimator_)
'''
for thresh in thresholds:
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit=True) # prefit에 대해 알아볼것 과제!
    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)
    selection_model = 
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%"%(thresh,select_x_train.shape[1],score*100))

'''