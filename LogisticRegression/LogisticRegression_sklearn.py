# Logistic Regression - sklearn 사용
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import pandas as pd

iris_data = load_iris()

#iris_data
x = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y = pd.DataFrame(iris_data.target, columns=['class'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

#x_train = torch.FloatTensor(x_train.values)
#x_test = torch.FloatTensor(x_test.values)
#y_train = torch.IntTensor(y_train.values)
#y_test = torch.FloatTensor(y_test.values)

y_train = y_train.values.ravel()   # 로지스틱 회귀를 할 때는 써주는 것이 에러가 발생하지 않고 좋다
#y_train = y_train.ravel()
#y_train.astype(int)

model = LogisticRegression(solver='saga', max_iter=2000)   # 옵션은 사용하지 않아도 됨
#model = LogisticRegression()

model.fit(x_train, y_train)
model.predict(x_test)
model.score(x_test, y_test)

print(model.coef_)
print(model.intercept_)
