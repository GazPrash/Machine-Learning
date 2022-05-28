import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from word2number import w2n

# Importing and cleaning data
hiring_data = pd.read_csv(
    "https://raw.githubusercontent.com/codebasics/py/master/ML/2_linear_reg_multivariate/Exercise/hiring.csv"
)
median_test_score = hiring_data["test_score(out of 10)"].median()
hiring_data["experience"] = hiring_data["experience"].fillna("zero")
hiring_data["test_score(out of 10)"] = hiring_data["test_score(out of 10)"].fillna(median_test_score)
hiring_data["experience"] = hiring_data["experience"].apply(lambda x : w2n.word_to_num(x))
# print(hiring_data)

# Ini. Model
model = linear_model.LinearRegression()
model.fit(
    hiring_data[["experience", "test_score(out of 10)", "interview_score(out of 10)"]].values,
    hiring_data["salary($)"].values
)

print(f"The predicted salary for this employee is : ${round(float(model.predict([[13, 8, 9]])), 2)}")