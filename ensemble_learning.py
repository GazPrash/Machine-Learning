# Ensemble Learning is all about dividing your dataset into n-smaller parts and then using n-different
# models to predict outcomes, then we take a majority vote.
# The motive of doing so ensures that we avoid the High-Varience problem, and multiple weak learning
# models can perform much better together.
# Ensemble Learning can be used for Regression or Class
# This methodology is called Bootstrap Aggeration

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

data = pd.read_csv(
    "https://raw.githubusercontent.com/codebasics/py/master/ML/19_Bagging/diabetes.csv"
)

features = data.drop("Outcome", axis= 1)
target = data["Outcome"]

scaler = MinMaxScaler()
features = scaler.fit_transform(features)
# print(features)

xtrain, xtest, ytrain, ytest = train_test_split(features, target.values)

model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
acc = model.score(xtest, ytest)


# Using Bagging Classifier 
model2 = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True
).fit(
    xtrain,
    ytrain
)

acc2 = model2.score(xtest, ytest)

print(acc2)












