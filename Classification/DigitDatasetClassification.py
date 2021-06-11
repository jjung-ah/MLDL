# digit dataset Classification

############ 데이터 확인 ################
%matplotlib inline
import matplotlib.pyplot as plt # 시각화를 위한 맷플롯립
from sklearn.datasets import load_digits
digits = load_digits() # 1,979개의 이미지 데이터 로드

print(digits.images[0])
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
print(digits.target[0])
print('전체 샘플의 수 : {}'.format(len(digits.images)))

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:5]):   # 5개의 샘플만 출력
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('sample: %i' % label)

for i in range(5):
  print(i,'번 인덱스 샘플의 레이블 : ',digits.target[i])

print(digits.data[0])   # 8 × 8 행렬이 아니라 64차원의 벡터로 저장된 것을 볼 수 있다.

X = digits.data # 이미지. 즉, 특성 행렬
Y = digits.target # 각 이미지에 대한 레이블


############# 분류기 만들기 ####################
import torch
import torch.nn as nn
from torch import optim

model = nn.Sequential(
    nn.Linear(64, 32),  # input_layer = 64, hidden_layer = 32
    nn.ReLU(),
    nn.Linear(32, 16),  # hidden_layer = 32, hidden_layer = 16
    nn.ReLU(),
    nn.Linear(16, 10)   # hidden_layer = 16, output_layer = 10
)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)

loss_fn = nn.CrossEntropyLoss() # 이 비용 함수는 소프트맥스 함수를 포함하고 있음.
optimizer = optim.Adam(model.parameters())

losses = []

for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(X) # forwar 연산
    loss = loss_fn(y_pred, Y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, 100, loss.item()
            ))

    losses.append(loss.item())

plt.plot(losses)


