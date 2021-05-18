# 다항회귀 - Polynomial Regression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd

boston_dataset = load_boston()
#print(boston_dataset.data.shape)
#print(boston_dataset.feature_names)
ploynomial_transformer = PolynomialFeatures(2)
polynomial_data = ploynomial_transformer.fit_transform(boston_dataset.data)
#print(polynomial_data.shape)
polynomial_feature_names = ploynomial_transformer.get_feature_names(boston_dataset.feature_names)
#print(polynomial_feature_names)
X = pd.DataFrame(polynomial_data, columns=polynomial_feature_names)
y = pd.DataFrame(boston_dataset.target, columns=['MEDV'])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

model = LinearRegression()
model.fit(x_train, y_train)
model.coef_
model.intercept_

y_test_prediction = model.predict(x_test)
mean_squared_error(y_test, y_test_prediction) ** 0.5