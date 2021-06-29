import pandas as pd

df = pd.read_csv('data/broadcast.csv', index_col=0)

# 코드를 작성하세요.
df.loc[2016, 'KBS']
df.loc[:, ['SBS', 'JTBC']]
df.loc[2012:2017, 'KBS':'SBS']
kbs_30 = (df.loc[:, 'KBS'] > 30)
df[kbs_30].loc[:, 'KBS']
compare_data = df['SBS'] < df['TV CHOSUN']
df.loc[compare_data].loc[:, ['SBS', 'TV CHOSUN']]
