import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv(
   "https://raw.githubusercontent.com/codebasics/py/master/DataScience/BangloreHomePrices/model/bengaluru_house_prices.csv"
)

# utility functions
def handleStrData(x:str):
    if "-" in x:
        try:
            out:float = (float(x.split()[0]) + float(x.split()[-1]))/2
        except Exception:
            return np.nan
    else :
        try:
            out:float = float(x)
        except Exception:
            return np.nan

    return out


# data cleansing and transformation
data["total_sqft"] = data["total_sqft"].apply(lambda x : handleStrData(x))
data.dropna(inplace= True)

target = data["price"]
features = data.drop(["availability", "society"], axis= 1)

area_type = pd.get_dummies(data["area_type"], drop_first=1)
location = pd.get_dummies(data["location"], drop_first=1)
classf_data = pd.concat([area_type, location], axis= 1)

features.drop(["location", "area_type"], axis= 1, inplace= True)
features = pd.concat([features, classf_data], axis=1)

features["size"] = features["size"].apply(lambda x : int(x.split(" ")[0]))

# preparing training-testing data
xtrain, xtest, ytrain, ytest = train_test_split(features.values, target.values, test_size=0.25)

# model ini

model = LinearRegression()
model.fit(xtrain, ytrain)
acc = model.score(xtest, ytest)

print(acc)