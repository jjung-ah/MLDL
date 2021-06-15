# XOR - 단층퍼셉트론 구현하기 (결론 : 불가능)

import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# XOR 게이트에 해당하는 입출력 정의
X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

# 1개 뉴런을 가지는 단층 퍼셉트론 구현
linear = nn.Linear(2, 1, bias=True)
sigmoid = nn.Sigmoid()
model = nn.Sequential(linear, sigmoid).to(device)

# 0 또는 1을 예측하는 이진 분류 문제이므로 비용 함수로는 크로스엔트로피 사용
# nn.BCELoss()는 이진 분류에서 사용하는 크로스엔트로피 함수
# 비용함수와 옵티마이저 정의
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

# 10,001번의 에폭 수행
for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)
    
    # 비용함수
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()
    
    if step % 1000 == 0:
        print(step, cost.item())

# 어느 순간 이후, 비용함수가 줄어들지 않는다. 이는 단층 퍼셉트론은 XOR 문제를 풀 수 없기 때문이다.




# 학습된 단층 퍼셉트론의 예측값 확인하기
with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(Y): ', Y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())

# 실제값은 0, 1, 1, 0임에도 예측값은 0, 0, 0, 0으로 문제를 풀지 못하는 모습을 보여줌