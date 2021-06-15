# Multiple Linear Regression - pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

torch.manual_seed(1)


boston_dataset = load_boston()
X = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
y = pd.DataFrame(boston_dataset.target, columns=['MEDV'])

# 데이터
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
# DataFrame --> numpy로 변환( .value 이용) --> numpy를 torch.tensor로 변환(torch.from_numpy 이용)
x_train = torch.FloatTensor(x_train.values)
x_test = torch.FloatTensor(x_test.values)
y_train = torch.FloatTensor(y_train.values)
y_test = torch.FloatTensor(y_test.values)


# 가중치 w와 편향 b 초기화
W = torch.zeros((13, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-6)
# 지정한 스텝 단위로 학습률에 감마를 곱해 학습률을 감소시키는 방식
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma= 0.92) 

nb_epochs = 1000000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train.matmul(W) + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    
    # scheduler 개선(lr개선)
    #scheduler.step(cost)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100000 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
    
    # 100번마다 로그 출력
    #if epoch % 100 == 0:
    #    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
    #        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    #    ))


 #print(x_train.shape)