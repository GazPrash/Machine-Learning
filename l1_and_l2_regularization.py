# L1 & L2 Regularization can help in maintaing a balance fit while training the model
# for more info : check out Fitting_data_possibilites.png

from pyexpat import model
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv(
    "https://raw.githubusercontent.com/codebasics/py/master/ML/16_regularization/Melbourne_housing_FULL.csv"
)

cols = [
    "Suburb",
    "Rooms",
    "Type",
    "Method",
    "SellerG",
    "Regionname",
    "Propertycount",
    "Distance",
    "CouncilArea",
    "Bedroom2",
    "Bathroom",
    "Car",
    "Landsize",
    "BuildingArea",
    "Price",
]

data = data[cols]
na_cols = ["Propertycount", "Bedroom2", "Bathroom", "Distance", "Car"]
data[na_cols] = data[na_cols].fillna(0)

data["Landsize"] = data["Landsize"].fillna(data["Landsize"].mean())
data["BuildingArea"] = data["BuildingArea"].fillna(data["BuildingArea"].mean())
# data["Price"] = data["Price"].fillna(data["Price"].mean())
data.dropna(inplace= True)
# print(data.isnull().sum())

data = pd.get_dummies(data, drop_first=1)

# print(data)

target = data.Price
features = data.drop(["Price"], axis = 1)

# print(target)

xtrain, xtest, ytrain, ytest = train_test_split(features.values, target.values, test_size=0.25)
model1 = LinearRegression()
model1.fit(xtrain, ytrain)
# print(model1.score(xtest, ytest))
# Only 54% accurate on test data
# print(model1.score(xtrain, ytrain))
# 67% acc on training data i.e not a balanced fit


# Lasso Regression is L1 Regularization
model2 = Lasso(alpha=50, max_iter=1e3, tol=1e-2)
model2.fit(xtrain, ytrain)
print(model2.score(xtest, ytest))