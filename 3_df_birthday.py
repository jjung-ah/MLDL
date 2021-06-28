import pandas as pd

# 코드를 작성하세요.
name = ['Taylor Swift', 'Aaron Sorkin', 'Harry Potter', 'Ji-Sung Park']
birthday = ['December 13, 1989', 'June 9, 1961', 'July 31, 1980', 'February 25, 1981']
occupation = ['Singer-songwriter', 'Screenwriter', 'Wizard', 'Footballer']

dataframe_list = {
    'name': name,
    'birthday': birthday,
    'occupation': occupation
}

df = pd.DataFrame(dataframe_list)
# 정답 출력
df