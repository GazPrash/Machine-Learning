# context : https://www.youtube.com/watch?v=9yl6-HEY7_s
# handling nominal categorical values which have no numerical/mathematical relation b/w them


from matplotlib.pyplot import axis
import pandas as pd
from sklearn import linear_model
from sklearn.compose import ColumnTransformer

data = pd.read_csv(
    "https://raw.githubusercontent.com/codebasics/py/master/ML/5_one_hot_encoding/homeprices.csv"
)

#  Pandas get_dummies method very useful for assigning values to nominal cat. data
# Alternatively we can use filtering and apply method directly on the df too.

dum = pd.get_dummies(data.town)
data = pd.concat([data, dum] ,axis="columns")

org_data = data.copy(deep = True)

# IMPORTANT : Dummy variable trap
# The Dummy variable trap is a scenario where there are attributes that are highly correlated (Multicollinear) 
# and one variable predicts the value of others. When we use one-hot encoding for handling the 
# categorical data, then one dummy variable (attribute) can be predicted with the help of other 
# dummy variables. Hence, one dummy variable is highly correlated with other dummy variables.
# Using all dummy variables for regression models leads to a dummy variable trap. 
# So, the regression models should be designed to exclude one dummy variable. 

# For Example – 
# Let’s consider the case of gender having two values male (0 or 1) and female (1 or 0). 
# Including both the dummy variable can cause redundancy because if a person is not male in such 
# case that person is a female, hence, we don’t need to use both the variables in regression models. 
# This will protect us from the dummy variable trap.

# for more info : https://www.geeksforgeeks.org/ml-dummy-variable-trap-in-regression-models/


data = (data.drop(["town", "robinsville"], axis=1))
# print(data)

# model ini & training
model = linear_model.LinearRegression()
model.fit(data[["area", "monroe township", "west windsor"]].values, data["price"].values)

pred = model.predict([[2500, 0, 1]])
# print(pred)

# Accuracy
acc = model.score(data[["area", "monroe township", "west windsor"]].values, data["price"].values)
# print(acc)


# One Hot Encoding using sklear preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

lencode = LabelEncoder()
org_data.town =  lencode.fit_transform(org_data["town"])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
    remainder='passthrough' 
)
X_values = ct.fit_transform(org_data[["town"]].values)
X_values = X_values[:, 1:]
print(X_values)
