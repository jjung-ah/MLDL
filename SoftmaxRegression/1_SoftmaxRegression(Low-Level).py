# 1. 소프트맥스 회귀 구현하기(로우-레벨)

import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import pandas as pd

iris_data = load_iris()
x = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y = pd.DataFrame(iris_data.target, columns=['class'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

x_train = torch.FloatTensor(x_train.values)
x_test = torch.FloatTensor(x_test.values)
y_train = torch.LongTensor(y_train.values)
y_test = torch.LongTensor(y_test.values)
#y_train = y_train.values.ravel()

# print(iris_data.DESCR)

# 데이터 shape 확인
print(x_train.shape)
print(y_train.shape)
# x_train의 각 샘플은 4개의 특성을 가지고 있으며, 총 120개의 샘플이 존재한다. 
# y_train은 각 샘플에 대한 레이블(class)인데, 여기서는 Iris-Setosa, Iris-Versicolour, Iris-Virginica의 값을 가지는 것으로 보아 총 3개의 클래스가 존재한다.

y_one_hot = torch.zeros(120, 3)
y_one_hot.scatter_(1, y_train, 1)
#y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)   # unsqueeze(1)할 필요가 없다. 이미 [120, 1]의 형태로 y_train의 shape이 만들어져있기 때문
print(y_one_hot.shape)



# y_train에서 원-핫 인코딩을 한 결과인 y_one_hot의 크기는 120x3이다. 
# 즉, W 행렬의 크기는 4x3이어야 한다.
# W와 b를 선언하고, 옵티마이러조는 경사하강법을 사용한다. 그리고 학습률을 1e-2로 설정한다. 

# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-2)

# F.softmax()와 torch.log()를 사용하여 가설과 비용 함수를 정의하고, 총 5,000번의 에포크를 수행한다.
nb_epochs = 5000
for epoch in range(nb_epochs + 1):

    # 가설
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1) 

    # 비용 함수
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 500 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))