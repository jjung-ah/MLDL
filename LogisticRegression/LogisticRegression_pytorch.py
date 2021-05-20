# Logistic Regression - pytorch 사용
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import pandas as pd

iris_data = load_iris()
x = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y = pd.DataFrame(iris_data.target, columns=['class'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

x_train = torch.FloatTensor(x_train.values)
x_test = torch.FloatTensor(x_test.values)
y_train = torch.FloatTensor(y_train.values)
y_test = torch.FloatTensor(y_test.values)
#y_train = y_train.values.ravel()

#print(x_train.shape)
#print(y_train.shape)

W = torch.zeros((4, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#hypothesis = 1/(1+torch.exp(-(x_train.matmul(W) + b)))
#print(hypothesis)
hypothesis = torch.sigmoid(x_train.matmul(W) + b)

#losses = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis))
#print(losses)
#cost = losses.mean()
losses = F.binary_cross_entropy(hypothesis, y_train)

W = torch.zeros((4, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([W, b], lr=1e-5)

nb_epochs = 5000
for epoch in range(nb_epochs+1):
    
    # Cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    cost = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)).mean()
    
    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))


hypothesis = torch.sigmoid(x_train.matmul(W) + b)
#print(hypothesis)
print(W)
print(b)
