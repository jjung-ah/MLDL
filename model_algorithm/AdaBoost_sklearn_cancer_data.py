# AdaBoost_sklearn_cancer_data

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

import pandas as pd

# 데이터 셋 불러 오기
cancer_data = load_breast_cancer()

# 챕터 1 유방암 데이터 준비하기 과제에서 쓴 코드를 갖고 오세요
X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = pd.DataFrame(cancer_data.target, columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
y_train = y_train.values.ravel()   # 경고메시지 출력 방지

# 코드를 쓰세요
model = AdaBoostClassifier(n_estimators=50, random_state=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = model.score(X_test, y_test)
#importances = model.feature_importances_
#indices_sorted = np.argsort(importances)

# 출력 코드
predictions, score
