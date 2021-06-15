# 2. 소프트맥스 회귀 구현하기(하이-레벨)
# 이제는 F.cross_entropy()를 사용하여 비용 함수를 구현한다.
# 주의할 점은 F.cross_entropy()는 그 자체로 소프트맥스 함수를 포함하고 있으므로 가설에서는 소프트맥스 함수를 사용할 필요가 없다.

import torch
import torch.nn as nn
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

y_one_hot = torch.zeros(120, 3)
y_one_hot.scatter_(1, y_train, 1)
#y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)   # unsqueeze(1)할 필요가 없다. 이미 [120, 1]의 형태로 y_train의 shape이 만들어져있기 때문
print(y_one_hot.shape)

# 모델 초기화
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 5000
for epoch in range(nb_epochs + 1):

    # Cost 계산
    z = x_train.matmul(W) + b
    #print(z.shape)
    #print(y_train.shape)
    #cost = F.cross_entropy(z, y_train)
    cost = F.cross_entropy(z, y_train.squeeze(1))  # y_train의 shape이 (120,1)이어서 (120,)으로 바꾸어 주기위해 squeeze(1) 사용

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 500 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))