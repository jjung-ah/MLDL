# k-fold cv
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import numpy as np 
import pandas as pd 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

iris_data = datasets.load_iris()

x = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
y = pd.DataFrame(iris_data.target, columns=['Class'])

logistic_model = LogisticRegression(max_iter=2000)
# k겹 교차검증을 해주는 함수
np.average(cross_val_score(logistic_model, x, y.values.ravel(), cv=5))


