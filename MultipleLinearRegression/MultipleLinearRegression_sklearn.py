# Multiple Linear Regression - sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd

boston_dataset = load_boston()

X = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
y = pd.DataFrame(boston_dataset.target, columns=['MEDV'])

# 데이터
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
# DataFrame --> numpy로 변환( .value 이용) --> numpy를 torch.tensor로 변환(torch.from_numpy 이용)
#x_train = torch.from_numpy(x_train.values)
#x_test = torch.from_numpy(x_test.values)
#y_train = torch.from_numpy(y_train.values)
#y_test = torch.from_numpy(y_test.values)

model = LinearRegression()
model.fit(x_train, y_train)
print(model.coef_)
print(model.intercept_)
y_test_predict = model.predict(x_test)
mean_squared_error(y_test, y_test_predict) ** 0.5
