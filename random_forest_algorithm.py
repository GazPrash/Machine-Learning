import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import seaborn as sns

# Random forest algorithm includes taking different samples and then producing different
# Decision Trees out of that (Hence the term of forest coming into play). The trees will come with
# their own different trees and we select out answer by taking a majority vote.

dig = load_digits()
data = dig.data
target = dig.target

xtrain, xtest, ytrain, ytest = train_test_split(data, target, test_size= 0.2)


# using default 10 n-estimators | 10 random trees
model = RandomForestClassifier(n_estimators= 40)
model.fit(xtrain, ytrain)
acc = model.score(xtest, ytest)


# Heatmap Truth vs Prediction

ypred = model.predict(xtest)
yorg = ytest

cm = confusion_matrix(ytest, ypred)
sns.heatmap(cm, annot = True)
plt.show()