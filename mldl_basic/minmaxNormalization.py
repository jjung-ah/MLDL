# Feature Scailing
# Min-max normalization

import pandas as pd
import numpy as np

from sklearn import preprocessing


NBA_FILE_PATH = 'E:\\MyData\\NBA_player_of_the_week.csv'
nba_player_of_the_week_df = pd.read_csv(NBA_FILE_PATH)

nba_player_of_the_week_df.head()
nba_player_of_the_week_df.describe()
height_weight_age_df = nba_player_of_the_week_df[['Height CM', 'Weight KG', 'Age']]
height_weight_age_df.head()

scaler = preprocessing.MinMaxScaler()
normalized_data = scaler.fit_transform(height_weight_age_df)
#normalized_data

normalized_df = pd.DataFrame(normalized_data, columns=['Height', 'Weight', 'Age'])
normalized_df.describe()

