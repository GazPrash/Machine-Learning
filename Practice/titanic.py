import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv(
    "https://raw.githubusercontent.com/codebasics/py/master/ML/9_decision_tree/Exercise/titanic.csv"
)
data.Age = data.Age.fillna(data.Age.median())

features = data[["Pclass", "Sex", "Age", "Fare"]]
target = data["Survived"]

n_sex = LabelEncoder()
features["n_sex"] = n_sex.fit_transform(features["Sex"])
features = features.drop(["Sex"], axis = 1)

print(features)

Xtrain, Xtest, ytrain, ytest = train_test_split(features.values, target.values, test_size= 0.25)
model = DecisionTreeClassifier()
model.fit(Xtrain, ytrain)

acc = model.score(Xtest ,ytest)

print(acc)
# 78.026% Accurate
