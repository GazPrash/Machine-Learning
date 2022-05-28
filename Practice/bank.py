import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

data = pd.read_csv("data/bank-full.csv")
data.dropna(inplace= True)
data["y"] = data["y"].apply(lambda x : 1 if x == "yes" else 0)
data["housing"] = data["housing"].apply(lambda x : 1 if x == "yes" else 0)
data["loan"] = data["loan"].apply(lambda x : 1 if x == "yes" else 0)

def set_edu(x:str)->int :
    if x == "unknown":
        return 0
    elif x == "primary":
        return 2
    elif x == "secondary":
        return 3
    else : return 4

data["education"] = data["education"].apply(lambda x : set_edu(x))

dummy = pd.get_dummies(data.marital)
data = pd.concat([data, dummy], axis=1)
data = data.drop(["marital", "divorced"],axis = 1)
# print(data)

X_train, X_test, y_train, y_test = train_test_split(data[["age", "education","married", "single", "balance", "housing", "loan"]].values,
                                                            data["y"].values,
                                                            test_size=0.2
                                                            )

model = linear_model.LogisticRegression(max_iter= 100000)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(acc)

print(model.predict([[32, 3, 1, 0, 15000, 0, 0]]))