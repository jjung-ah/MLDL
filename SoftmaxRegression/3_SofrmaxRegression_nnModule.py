# 3. 소프트맥스 회귀 nn.Module로 구현하기
# 이번에는 nn.Module로 소프트맥스 회귀를 구현한다. 선형회귀에서 구현에 사용했던 nn.Linear()를 사용한다.
# output_dim이 1이었던 선형회귀때와 달리 output_dim은 이제 클래스의 개수여야 한다.

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


# 모델을 선언 및 초기화. 4개의 특성을 가지고 3개의 클래스로 분류. input_dim=4, output_dim=3
model = nn.Linear(4, 3)

# 아래에서 F.cross_entropy()를 사용할 것이므로 따로 소프트맥스 함수를 가설에 정의하지 않는다.

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 5000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train.squeeze(1))   # y_train의 shape이 (120,1)이어서 (120,)으로 바꾸어 주기위해 squeeze(1) 사용

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 500 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))