# 필요한 도구 임포트
import numpy as np

# numpy 임의성 조절
np.random.seed(42)

def initialize_parameters(neurons_per_layer):
    """신경망의 가중치와 편향을 초기화해주는 함수"""
    L = len(neurons_per_layer)- 1  # 층 개수 저장
    parameters = {}
    
        # 1층 부터 L층까지 돌면서 가중치와 편향 초기화
    for l in range(1, L+1):
        parameters['W' + str(l)] = np.sqrt(1/neurons_per_layer[l])*np.random.randn(neurons_per_layer[l], neurons_per_layer[l-1])
        parameters['b' + str(l)] = np.sqrt(1/neurons_per_layer[l])*np.random.randn(neurons_per_layer[l])
        
    return parameters

# 실행 코드
neurons_per_layer = [10, 5, 5, 3]
initialize_parameters(neurons_per_layer)

