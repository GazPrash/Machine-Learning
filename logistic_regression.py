import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# Getting dataset
# data = pd.read_csv(
#     "https://raw.githubusercontent.com/codebasics/py/master/ML/7_logistic_reg/Exercise/HR_comma_sep.csv"
# )
# data.to_csv("data/hr_analytics.csv")

data = pd.read_csv("data/hr_analytics.csv")

# Exercise
# Download employee retention dataset from here: https://www.kaggle.com/giripujar/hr-analytics.

# Now do some exploratory data analysis to figure out which variables have direct and clear impact on 
# employee retention (i.e. whether they leave the company or continue to work)
#
# Plot bar charts showing impact of employee salaries on retention
# Plot bar charts showing corelation between department and employee retention
# Now build logistic regression model using variables that were narrowed down in step 1
# Measure the accuracy of the model


left = data[data["left"] == 1]
retained = data[data["left"] == 0]

l = left["salary"].value_counts()
r = retained["salary"].value_counts()

fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(l)
axes[0, 1].plot(r)
# plt.show()

# 2nd Question can be done in the same way
def salary_settler(x:str):
    if x == "low":
        return 1
    elif x == "medium":
        return 2
    else : return 3

data["salary"] = data["salary"].apply(lambda x : salary_settler(x))
X_train, X_test, y_train, y_test = train_test_split(data[["satisfaction_level",
                                                          "promotion_last_5years",
                                                          "salary",
                                                          "average_montly_hours",
                                                          "Work_accident"
                                                          ]].values,
                                                          data["left"].values,
                                                          test_size= 0.2
                                                        )

model = linear_model.LogisticRegression()
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print(acc)
# print(model.predict_proba([[0.95, 1, salary_settler("medium"), 0]]))

def predict_employee_retention(sats_lvl, promotion, salary, hours_invested, accident_counts):
    parameter_list = [sats_lvl, promotion, salary_settler(salary), hours_invested, accident_counts]
    model_prob = model.predict_proba([parameter_list])
    if model.predict([parameter_list]):
        print(f"The employee may end up leaving. Probablity Status : [{model_prob[0][0]}]")
        return

    print(f"The employee is not leaving. Probablity Status : [{model_prob[0][0]}]")

predict_employee_retention(0.05, 1, "low", 932, 1)
