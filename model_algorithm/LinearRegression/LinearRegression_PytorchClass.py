import torch
import torch.nn
import torch.nn.functional as F

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

torch.manual_seed(1)

#model = nn.Linear(1,1)


class LinearRegressionModel(nn.Module):   # torch.nn.Module을 상속받는 파이썬 클래스
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)   # 단순 선형 회귀이므로 input_dim=1, output_dim=1
        
    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

boston_dataset = load_boston()
X = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
y = pd.DataFrame(boston_dataset.target, columns=['MEDV'])
x = X[['AGE']]

# 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
# DataFrame --> numpy로 변환( .value 이용) --> numpy를 torch.tensor로 변환(torch.from_numpy 이용)
x_train = torch.FloatTensor(x_train.values)
x_test = torch.FloatTensor(x_test.values)
y_train = torch.FloatTensor(y_train.values)
y_test = torch.FloatTensor(y_test.values)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

nb_epochs = 200000
for epoch in range(nb_epochs+1):
    
    # H(x) 계산
    prediction = model(x_train)
    
    # cost 계산
    cost = F.mse_loss(prediction, y_train)
    
    # cost로  H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()
    
    if epoch % 20000 == 0:
        print('Epoch {:4d}/{}  Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))
