# Pytorch 활용
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

torch.manual_seed(1)

boston_dtaseet = load_boston()
X = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
#print(X.shape)
y = pd.DataFrame(boston_dataset.target, columns=['MEDV'])
#print(y.shape)
x = X[['AGE']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

'''
# DataFrame --> numpy로 변환( .value 이용) --> numpy를 torch.tensor로 변환(torch.from_numpy 이용)
x_train = torch.from_numpy(x_train.values)
x_test = torch.from_numpy(x_test.values)
y_train = torch.from_numpy(y_train.values)
y_test = torch.from_numpy(y_test.values)

# y = 0*x + 0
W = torch.zeros(1, requires_grad=True)
#print(W)   # tensor([0.], requires_grad=True)
b = torch.zeros(1, requires_grad=True)
#print(b)   # tensor([0.], requires_grad=True)

# 가설함수 H(x) = W*x + b
hypothesis = x_train * W + b
#print(hypothesis)

# cost function인 평균제곱오차(MeanSquareError)
cost = torch.mean((hypothesis - y_train) ** 2)
print(cost)

# 경사하강법 구현
optimizer = optim.SGD([W, b], lr=0.01)

# gradient를 0으로 초기화
optimizer.zero_grad()
# 비용함수를 미분하여 gradient를 계산
cost.backward()
# W와 b를 업데이트
optimizer.step()
'''


# 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
# DataFrame --> numpy로 변환( .value 이용) --> numpy를 torch.tensor로 변환(torch.from_numpy 이용)
x_train = torch.from_numpy(x_train.values)
x_test = torch.from_numpy(x_test.values)
y_train = torch.from_numpy(y_train.values)
y_test = torch.from_numpy(y_test.values)

# 모델 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.00001)


nb_epochs = 10 
for epoch in range(nb_epochs + 1):
    
    # H(x) 계산
    hypothesis = x_train * W + b
    
    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 1 == 0:
        print('Epoch {:4d}/{} W: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
        epoch, nb_epochs, W.item(), b.item(), cost.item() ))