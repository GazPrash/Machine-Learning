import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv(
    "https://raw.githubusercontent.com/codebasics/py/master/ML/9_decision_tree/salaries.csv"
)

# print(data)

features = data.drop(["salary_more_then_100k"], axis= 1)

n_company = LabelEncoder()
n_degree = LabelEncoder()
n_job = LabelEncoder()

features["n_company"] = (n_company.fit_transform(features["company"]))
features["n_job"] = (n_company.fit_transform(features["job"]))
features["n_degree"] = (n_company.fit_transform(features["degree"]))

input_features = features[["n_company", "n_degree", "n_job"]]

# print(input_features)

input_traget = data["salary_more_then_100k"]

Xtrain, Xtest, ytrain, ytest = train_test_split(input_features.values, input_traget.values, test_size=0.3)
model = DecisionTreeClassifier()
model.fit(Xtrain, ytrain)
acc = model.score(Xtest, ytest)

print(acc)