import pandas as pd

df1 = pd.read_csv('data/body_imperial1.csv', index_col=0)
df2 = pd.read_csv('data/body_imperial2.csv', index_col=0)

# 코드를 작성하세요.
df1.iloc[[1], [1]] = 200
df1.drop(21, axis='index', inplace=True)
df1.loc[20] = [70, 200]


# 코드를 작성하세요.
df2.iloc[[1], [1]] = 200
df2.drop(21, axis='index', inplace=True)
df2.loc[20] = [70, 200]


df1
df2