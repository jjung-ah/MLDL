# Multiple LinearRegression - pytorch Class
import torch
import torch.nn
import torch.nn.functional as F

torch.manual_seed(1)

#model = nn.Linear(13,1)

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

import pandas as pd

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


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(13,1)
        
    def forward(self, x):
        return self.linear(x)


model = MultivariateLinearRegressionModel()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

nb_epochs = 1000000
for epoch in range(nb_epochs+1):
    
    # H(x) 계산
    prediction = model(x_train)
    
    # cost 계산
    cost = F.mse_loss(prediction, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100000 == 0:
        print('Epoch {:4d}/{}  Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))


y_prediction = model(x_test)
F.mse_loss(y_test, y_prediction)
print(list(model.parameters()))