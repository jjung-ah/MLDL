import pandas as pd

df = pd.read_csv('https://github.com/codeit-courses/data-science/raw/master/Puzzle_before.csv')

# 코드를 작성하세요.
df['A'] = df['A'] * 2
df[df.iloc[:, 1:5] < 80] = 0
df[df.iloc[:, 1:5] >= 80] = 1
df.iloc[[2], [5]] = 99

# 정답 출력
df