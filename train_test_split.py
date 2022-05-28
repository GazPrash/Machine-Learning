import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


data = pd.read_csv(
    "https://raw.githubusercontent.com/codebasics/py/master/ML/6_train_test_split/carprices.csv"
)

# Train-Test split method is used 
# in machine learning projects to split available dataset into training and test set. 
# This way you can train and test on separate datasets. When you test your model using 
# dataset that model didn't see during training phase, it will give you better idea on 
# the accuracy of a model. 

X = data[["Mileage", "Age(yrs)"]].values
y = data["Sell Price($)"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)

print(model.predict(X_test)," and the acc is : ", acc)