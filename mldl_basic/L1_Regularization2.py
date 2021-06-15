from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt

import numpy as np
import pandas as pd

# 데이터 파일 경로 정의
INSURANCE_FILE_PATH = './datasets/insurance.csv'

insurance_df = pd.read_csv(INSURANCE_FILE_PATH)  # 데이터를 pandas dataframe으로 갖고 온다 (insurance_df.head()를 사용해서 데이터를 한 번 살펴보세요!)
insurance_df = pd.get_dummies(data=insurance_df, columns=['sex', 'smoker', 'region'])  # 필요한 열들에 One-hot Encoding을 해준다
#print(insurance_df)
# 입력 변수 데이터를 따로 새로운 dataframe에 저장
X = insurance_df.drop(['charges'], axis=1)
#print(X.shape)

polynomial_transformer = PolynomialFeatures(4)  # 4 차항 변형기를 정의
polynomial_features = polynomial_transformer.fit_transform(X.values)  #  4차 항 변수로 변환
#print(polynomial_features.shape)

features = polynomial_transformer.get_feature_names(X.columns)  # 새로운 변수 이름들 생성

X = pd.DataFrame(polynomial_features, columns=features)  # 다항 입력 변수를 dataframe으로 만들어 준다
y = insurance_df[['charges']]  # 목표 변수 정의

# 여기 코드를 쓰세요
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

model = Lasso(alpha=1, max_iter=2000, normalize=True)   # L1 정규화한 모델
#model = Ridge(alpha=0.001, max_iter=1000, normalize=True)   # L2 정규화한 모델
model.fit(X_train, y_train)

y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)

# 체점용 코드
mse = mean_squared_error(y_train, y_train_predict)

print("training set에서 성능")
print("-----------------------")
print(f'오차: {sqrt(mse)}')

mse = mean_squared_error(y_test, y_test_predict)

print("testing set에서 성능")
print("-----------------------")
print(f'오차: {sqrt(mse)}')
