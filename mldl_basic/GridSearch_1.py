# Grid Search
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from math import sqrt

import numpy as np 
import pandas as pd 

# 데이터 준비
ADMISSION_FILE_PATH = '../datasets/admission_data.csv'
admission_df = pd.read_csv(ADMISSION_FILE_PATH)

X = admission_df.drop('Chance of Admit ', axis=1)

polynomial_transformer = PolynomialFeatures(2)   # 2차식 변형기를 정의한다
polynomial_features = polynomial_transformer.fit_transform(X.values)

features = polynomial_transformer.get_feature_name(X.columns)

X = pd.DataFrame(polynomial_features, columns=features)
y = admission_df(['Chance of Admit '])

# 최적화할 파라미터의 딕셔너리
hyper_parameter = {
    'alpha': [0.01, 0.1, 1, 10],
    'max_iter': [100, 500, 1000, 1500, 2000]
}

lasso_model = Lasso()
hyper_parameter_tuner = GridSearchCV(lasso_model, hyper_parameter, cv=5)
hyper_parameter_tuner.fit(X, y)

hyper_parameter_tuner.best_params_