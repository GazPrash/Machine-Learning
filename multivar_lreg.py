import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# dependent var = (m1 * inde) + (m2 * inde) + (m3 * inde) + ... + b
# m1, m2, m3... are the coefs_ and they can help in relating the dep. var to the inde. ones
# independent vars are also called features & b is the intercept

hp = pd.read_csv("https://raw.githubusercontent.com/codebasics/py/master/ML/2_linear_reg_multivariate/homeprices.csv")

# Cleaning data
bed_median = float(hp["bedrooms"].median())
hp["bedrooms"] = hp["bedrooms"].apply(lambda x : bed_median if (math.isnan(x) == True) else x)

# Initializing & Training Model
model = linear_model.LinearRegression()
model.fit(hp[["area", "bedrooms", "age"]].values, hp.price.values)


# Predicting Dependent variable using multiple Inde vars
pred_price = model.predict([[4200, 5, 0.5]])
print(pred_price)

# 7200 * coef1 + 8 * coef2 + 34*coef3 + intercept