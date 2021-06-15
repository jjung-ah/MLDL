# 2. pytorch로 softmax 비용함수 구현하기 (High-Level)

# 2-(1). F.softmax() + torch.log() = F.log_softmax()
# Low level
torch.log(F.softmax(z, dim=1))

# High level
F.log_softmax(z, dim=1)



# 2-(2). F.log_softmax() + F.nll_loss() = F.cross_entropy()
# Low level
# 첫번째 수식
(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()

# 두번째 수식
(y_one_hot * -F.log_softmax(z, dim=1)).sum(dim=1).mean()

# High level
# 세번째 수식
F.nll_loss(F.log_softmax(z, dim=1), y)

# 여기서 nll이란, Negative Log Likelihood의 약자이다. 위에서 nll_loss는 F.log_softmax()를 수행한 후에 남은 수식들을 수행한다. 
# 이를 더 간단히 하면 다음과 같이 사용할 수 있다. F.cross_entropy()는 F.log_softmax()와 F.nll_loss()를 포함하고 있다. 

# 네번째 수식
F.cross_entropy(z, y)

