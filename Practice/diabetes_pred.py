import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes


# features = load_diabetes().feature_names
# target = load_diabetes().target
# # print(load_diabetes().keys())

# print(diabetes.frame)

diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)

print(acc)