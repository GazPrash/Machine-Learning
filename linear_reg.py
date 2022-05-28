import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# |       /
# |      /
# |     /
# |    /
# |   /    * line is drawn to minimize the delta (dist from scattered points nearby to assumed line)
# |  /     * dependedt_variable = slope * independent_variable + intercept
# | /      * line can be predicted by min. mean sq error
# |/
# ---------------------

# When we train data using the linear regression model in sklear, the fit func requires first two parameters
# as X & y, where X should be in the format of a 2D array, we can pass X as a dataframe but that gives us
# warnings as the dataframe can have undefined labels, hence dataframe should be passed as follows : 
# 
# 
#   X = df[["Independent Variable"]].values  && y = df["Dependent Variable"].values

# print(estmtr.predict([[5550]])) (use 2D Arrays for predict func too)



home_prices = pd.read_csv(
                "data/homeprices.csv"
            )
            
reg_model = linear_model.LinearRegression()
estmtr = reg_model.fit(home_prices[["area"]].values, home_prices.price.values)


areas = pd.Series(
    [
        1000,
        1500,
        2300,
        3540,
        4120,
        4560,
        5490,
        3460,
        4750,
        2300,
        9000,
        8600,
        7100,
    ]
)

rstate_data = pd.DataFrame()
rstate_data["Areas"] = areas

def predict_prices(df):
    df["Prices"] = estmtr.predict(df.values)
    return df


rstate_data =  predict_prices(rstate_data[["Areas"]])
print(rstate_data)







