# 5. 소프트맥스 회귀로 MNIST 데이터 분류하기

# 추가된 코드
!wget www.di.ens.fr/~lelarge/MNIST.tar.gz
!tar -zxvf MNIST.tar.gz
# 추가된 코드

import random
import warnings
warnings.filterwarnings(action="ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

'''
import numpy as np

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random
'''

# 현재 환경에서 GPU 연산이 가능하다면 GPU 연산을 하고, 그렇지 않다면 CPU 연산을 하도록 합니다.
USE_CUDA = torch.cuda.is_available()   # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu")
print("다음 기기로 학습합니다: ", device)

# 랜덤시드 고정
# for repreducibility
random.seed(777)
torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

# hyperparameters
training_epochs = 15
batch_size = 100


# torchvision.datasets.dsets.MNIST를 사용하여 MNIST 데이터셋을 불러올 수 있다
# MNIST dataset
mnist_train = MNIST(root='./',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

mnist_test = MNIST(root='./',
                        train=False,
                        transform=transforms.ToTensor(),
                        download=True)


# dataset loader
data_loader = DataLoader(dataset=mnist_train,
                                          batch_size=batch_size, # 배치 크기는 100
                                          shuffle=True,
                                          drop_last=True)

# MNIST data image of shape 28 * 28 = 784
#  input_dim은 784이고, output_dim은 10입니다.
linear = nn.Linear(784, 10, bias=True).to(device)

# 비용 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss().to(device) # 내부적으로 소프트맥스 함수를 포함하고 있음.
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)


for epoch in range(training_epochs): # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        # 배치 크기가 100이므로 아래의 연산에서 X는 (100, 784)의 텐서가 된다.
        # X는 호출될 때는 (배치 크기 × 1 × 28 × 28)의 크기를 가지지만, view를 통해서 (배치 크기 × 784)의 크기로 변환됩니다.
        X = X.view(-1, 28 * 28).to(device)
        # 레이블은 원-핫 인코딩이 된 상태가 아니라 0 ~ 9의 정수.
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
