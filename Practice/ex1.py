import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_csv("../data/canada_per_capita_income.csv")

model = linear_model.LinearRegression()
model.fit(data[["year"]].values, data.pci.values)

# Predict the Per Capita income by the year 2024

income = float(model.predict([[2024]]))
print(f"Inclome will be : ${income}")
ind= np.arange(2017, 2031)
ind = ind.reshape(-1, 1)
# ya toh array-like object ka dimension 2 rakho ya direct dataframe (data[["Variable_Name"]]) use karo 


plt.plot(ind, model.predict(ind))
plt.show()