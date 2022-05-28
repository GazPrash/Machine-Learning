import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


data = pd.read_csv(
    "https://raw.githubusercontent.com/codebasics/py/master/ML/5_one_hot_encoding/Exercise/carprices.csv"
)

# Questions : 
# At the same level as this notebook on github, there is an Exercise folder that contains carprices.csv. 
# This file has car sell prices for 3 different models. 
# First plot data points on a scatter plot chart to see if linear regression model can be applied. 
# If yes, then build a model that can answer following questions,

# 1) Predict price of a mercedez benz that is 4 yr old with mileage 45000
# 2) Predict price of a BMW X5 that is 7 yr old with mileage 86000
# 3) Tell me the score (accuracy) of your model. (Hint: use LinearRegression().score())

# Solution :  
# The scatter plots prove that yes there is indeed a relation b/w these features.
plt.scatter(data["Mileage"], data["Sell Price($)"], marker = 'H')

# getting data ready
dummy = pd.get_dummies(data["Car Model"])
data = pd.concat([data, dummy], axis = 1)
data.drop(["Car Model", "Mercedez Benz C class"], axis=1, inplace= True)

# Ini & Training Model
model = linear_model.LinearRegression()
X = data[["Mileage", "Age(yrs)", "Audi A5", "BMW X5"]].values
y = data["Sell Price($)"].values
model.fit(X, y)


print(f"Price of that benz will be : {model.predict([[45000, 4, 0, 0]])}")
print(f"Price of that BMW X5 will be : {model.predict([[86000, 7, 0, 1]])}")
print(f"Model accuracy : {model.score(X, y)}")