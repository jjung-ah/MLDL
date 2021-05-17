# sklearn 활용
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd

boston_dataset = load_boston()
#print(boston_dataset.DESCR)
#print(boston_dataset.feature_names)
#print(boston_dataset.data.shape)
#print(boston_dataset.target.shape)

x = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
x = x[['AGE']]
y = pd.DataFrame(boston_dataset.target, columns=['MEDV'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

model = LinearRegression()
model.fit(x_train, y_train)
print(model.coef_ )
print(model.intercept_)
y_test_prediction = model.predict(x_test)
print(mean_squared_error(y_test, y_test_prediction) ** 0.5)
