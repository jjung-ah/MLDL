# Torchtext tutorial
# !pip install torchtext


############### 1.훈련 데이터와 테스트 데이터로 분리하기 ##################
import urllib.request
import pandas as pd

# 우선 인터넷에서 IMDB 리뷰 데이터를 다운로드 받습니다.
urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")

df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')
df.head()
print('전체 샘플의 개수 : {}'.format(len(df)))

# 전체 샘플의 개수는 50,000개입니다. 25,000개씩 분리하여 훈련 데이터와 테스트 데이터로 분리해보겠습니다.
train_df = df[:25000]
test_df = df[25000:]

train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)
# index=False를 하면 인덱스를 저장하지 않습니다.



############### 2. 필드 정의하기(torchtext.data) #########################
#from torchtext import data # torchtext.data 임포트
from torchtext.legacy import data   # torchtext.data.Field-> torchtext.legacy.data.Field 로 변경됨

# 필드 정의
TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True,
                  fix_length=20)

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   batch_first=False,
                   is_target=True)


############### 3. 데이터셋 만들기 #######################################
#from torchtext.data import TabularDataset
from torchtext.legacy.data import TabularDataset

train_data, test_data = TabularDataset.splits(
        path='.', train='train_data.csv', test='test_data.csv', format='csv',
        fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

print('훈련 샘플의 개수 : {}'.format(len(train_data)))
print('테스트 샘플의 개수 : {}'.format(len(test_data)))

print(vars(train_data[0]))   # vars()를 통해서 주어진 인덱스의 샘플을 확인
print(train_data.fields.items())   # 필드 구성 확인



############### 4. 단어 집합(Vocabulary) 만들기 ############################
# 정의한 필드에 .build_vocab() 도구를 사용하면 단어 집합을 생성
TEXT.build_vocab(train_data, min_freq=10, max_size=10000)
print('단어 집합의 크기 : {}'.format(len(TEXT.vocab)))
print(TEXT.vocab.stoi)   # 생성된 단어 집합 내의 단어들은 .stoi를 통해서 확인 가능


############### 5. 토치텍스트의 데이터로더 만들기 ###########################
from torchtext.legacy.data import Iterator

batch_size = 5

train_loader = Iterator(dataset=train_data, batch_size = batch_size)
test_loader = Iterator(dataset=test_data, batch_size = batch_size)
print('훈련 데이터의 미니 배치 수 : {}'.format(len(train_loader)))
print('테스트 데이터의 미니 배치 수 : {}'.format(len(test_loader)))

batch = next(iter(train_loader)) # 첫번째 미니배치
print(type(batch))
print(batch.text)






