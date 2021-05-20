# Softmax 회귀 - 비용함수 구하기
# 1. pytorch로 softmax 비용함수 구현하기 (Low-Level)

import torch 
import torch.nn.functional as F

torch.manual_seed(1)

z = torch.FloatTensor([1, 2, 3])
# 이 텐서를 소프트맥스의 입력으로 사용하고, 결과 확인

hypothesis = F.softmax(z, dim=0)
print(hypothesis)
# 3개의 원소의 값이 0과 1사이의 값을 가지는 벡터로 변환된 것을 볼 수 있다. 

hypothesis.sum()
# 총 원소의 합은 1이다. 

z = torch.rand(3, 5, requires_grad=True)
# 비용함수를 구현하기 위해 임의의 3x5 행렬의 크기를 가진 텐서를 생성
# 이 텐서에 소프트맥스 함수 적용
# 각 샘플에 대해서 소프트맥스 함수를 적용하여야 하므로 두번째 차원에 대해서 소프트맥스 함수를 적용한다는 의미에서 dim=1을 써준다.

hypothesis = F.softmax(z, dim=1)
print(hypothesis)
# 각 행의 원소들의 합은 1이 되는 텐서로 변환되었다. 소프트맥스 함수의 출력값은 결국 예측값이다.
# 즉, 위의 텐서는 3개 샘플에 대해서 5개의 클래스 중 어떤 클래스가 정답인지를 예측한 결과이다. 

# 각 샘플에 대해 임의의 레이블을 생성
y = torch.randint(5, (3,)).long()
print(y)

# 이제 각 레이블에 대해 원-핫 인코딩을 수행
y_one_hot = torch.zeros_like(hypothesis)   # 모든 원소가 0의 값을 가진 3x5 텐서 생성
y_one_hot.scatter_(1, y.unsqueeze(1), 1)   # scatter_(축, 새로 나타낼 인덱스, 새로 저장할 입력값)


# 위의 연산에서 어떻게 원-핫 인코딩이 수행되었는지 보면, 
# 우선, torch.zeros_like(hypothesis)를 통해 모든 원소가 0의 값을 가진 3x5텐서를 만들고 이 텐서는 y_one_hot에 저장이 된 상태이다.
# y.unsqueeze(1)을 하면 (3,)의 크기를 가졌던 y 텐서는 (3x1)텐서가 된다. (아래와 같이.)
print(y.unsqueeze(1))

# 그리고 scatter의 첫번째 인자로 dim=1에 대해 수행하라고 알려주고, 세번째 인자에 숫자 1을 넣어주므로서 두번째 인자인 y.unsqueeze(1)이 알려주는 위치에 숫자 1을 넣도록 한다.
# 앞서 텐서 조작하기 2챕터에서 연산 뒤에 _를 붙이면 In-place Operation(덮어쓰기 연산)임을 배운 바 있다. 이에 따라 y_one_hot의 최종 결과는 결국 아래와 같다.
print(y_one_hot)

# 소프트맥스 회귀의 비용함수는 다음과 같다.
# 소프트맥스 회귀에서는 비용함수로 크로스 엔트로피 함수를 사용한다. 
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)
