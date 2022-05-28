# PCA helps in Reducing Dimensions or Feature-Dependencies in Models i.e removing features that are not
# important or don't affect the model signficantly or at all.
# The result will be faster training & inference. Also easy Data Visualization


import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


dig = load_digits()
data = dig.data
target = dig.target

scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data))

xtrain, xtest, ytrain, ytest = train_test_split(data.values, target, test_size= 0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(xtrain, ytrain)
acc = model.score(xtest, ytest)

# print(acc) 98% - With 100 n_estimators

pca_obj = PCA(0.95)
eff_data = pca_obj.fit_transform(data.values)  # target variable ignored

xtrain_dash, xtest_dash, ytrain_dash, ytest_dash = train_test_split(eff_data, target, test_size= 0.2)

model.fit(xtrain_dash, ytrain_dash)
acc2 = model.score(xtest_dash, ytest_dash)

# print(acc2) Acc after dropping 33 Cols : 97.22%