# L1 정규화 모델을 이용한 과적합 문제 해결하기
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklear.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

from math import sqrt

import numpy as np 
import pandas as pd 

ADMISSION_FILE_PATH = '../datasets/admission_data.csv'
admission_df = pd.read_csv(ADMISSION_FILE_PATH).drop('Serial No.', axis=1)

admission_df.head()

X = admission_df.drop(['Chance of Admit '], axis=1)

polynomial_transformer = PolynomialFeatures(6)
polynomial_features = polynomial_transformer.fit_transform(X.values)
features = polynomial_transformer.get_feature_names(X.columns)

X = pd.DataFrame(polynomial_features, columns=features)
X.head()

y = admission_df[['Chance of Admit ']]
y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

model = Lasso(alpha=0.001, max_iter=1000, normalize=True)   # L1 정규화한 모델
#model = Ridge(alpha=0.001, max_iter=1000, normalize=True)   # L2 정규화한 모델
model.fit(X_train, y_train)

y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)

mse = mean_squared_error(y_train, y_train_predict)

print("training set에서의 성능")
print("-----------------------")
print(sqrt(mse))

mse = mean_squared_error(y_test, y_test_predict)

print("test set에서의 성능")
print("-----------------------")
print(sqrt(mse))
