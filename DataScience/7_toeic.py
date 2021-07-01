import pandas as pd
    
df = pd.read_csv('data/toeic.csv')

# 코드를 작성하세요.
hapbul = (df['LC']>=250) & (df['RC']>=250) & (df['LC'] + df['RC'] >= 600)
df['합격 여부'] = hapbul

# 정답 출력
df