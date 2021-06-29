import pandas as pd

samsong_df = pd.read_csv('data/samsong.csv')
hyundee_df = pd.read_csv('data/hyundee.csv')

# 코드를 작성하세요.
df_sdic = {'day': samsong_df.loc[:, '요일'], 'samsong': samsong_df.loc[:, '문화생활비']}
df_hdic = {'day': hyundee_df.loc[:, '요일'], 'hyundee': hyundee_df.loc[:, '문화생활비']}
df_s = pd.DataFrame(df_sdic)
df_h = pd.DataFrame(df_hdic)
df = pd.merge(df_s, df_h, on='day', how='outer')
df