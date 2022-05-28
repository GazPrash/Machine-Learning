import pandas as pd
import numpy as np
from sklearn import linear_model
from math import isclose


def gradient_descent(x, y):
    data_len = len(x)
    learning_rate = 0.0002
    iterations = 0
    threshold_cost_value = -1

    # starting values for m & b (==0)
    coef = 0
    intercept = 0

    while 1:
        y_pred = coef * x + intercept
        mse = (1/data_len) * (sum([i*i for i in (y - y_pred)]))
        m_deriv = -(2/data_len) * sum(x*(y - y_pred))
        b_deriv = -(2/data_len) * sum(y - y_pred)
        coef -= (learning_rate * m_deriv)
        intercept -= (learning_rate * b_deriv)

        if isclose(threshold_cost_value, mse, rel_tol=1e-20):
            print(f"Breakpoint Reached | No. of iterations = {iterations}")
            print(f"Value of coef : {coef} & Value of intercept {intercept}")
            break
        threshold_cost_value = mse
        iterations += 1


data = pd.read_csv(
    "https://raw.githubusercontent.com/codebasics/py/master/ML/3_gradient_descent/Exercise/test_scores.csv"    
)

# print(data)

arr1 = np.array(data.math)
arr2 = np.array(data.cs)

# gradient_descent(arr1, arr2)
# * OUTPUT *
# Breakpoint Reached | No. of iterations = 415533
# Value of coef : 1.0177381667350405 & Value of intercept 1.9150826165722297

# now comparing obtianed values with sklearn model
model = linear_model.LinearRegression()
model.fit(data[["math"]].values, data["cs"].values)

print(model.coef_, model.intercept_)
# * OUTPUT *
# [1.01773624] 1.9152193111568891

print(model.predict([[99], [12], [50]]))