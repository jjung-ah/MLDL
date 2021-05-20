# Logistic Regression - nn.Module
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

model = nn.Sequential(
    nn.Linear(4, 1),
    nn.Sigmoid()
)

model(x_train)

optimizer = optim.SGD(model.parameters(), lr=1e-7)

nb_epochs = 100000
for epoch in range(nb_epochs+1):
    
    # H(x) 계산
    hypothesis = model(x_train)
    
    # Cost 계산
    cost = F.binary_cross_entropy(hypothesis, y_train)
    
    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 10000 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])   # 예측값이 0.5를 넘으면 True로 간주
        #print(prediction)
        correct_prediction = prediction.float() == y_train   # 실제값과 일치하는 경우만 True로 간주
        #print('correct_prediction: ', len(correct_prediction))
        #print(correct_prediction.sum().item())
        #accuracy = correct_prediction.sum().item() / len(correct_prediction)   # 정확도를 계산 # 결과가 이상하다(점점 정확도 낮아짐)
        accuracy = prediction.sum().item() / len(correct_prediction)
        
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(
            epoch, nb_epochs, cost.item(), accuracy*100,
        ))


print(list(model.parameters()))