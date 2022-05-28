from cProfile import label
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Support vector machine draws a hyper plane in n-dimensional space such that it maximizes
# the margin b/w different classsification groups 
# (Example : like a line seprating two diffrent scatter groups on a 2d plot or a plane doing the
# the same thing in a 3d enviroment.)

# Watch : https://www.youtube.com/watch?v=FB5EdxAGxQg&list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw&index=11
# for in-depth info about Gamma, Regularization & Kernal

iris = load_iris()
target_list = list(iris.target_names)

features = pd.DataFrame(iris.data, columns= iris.feature_names)
target = pd.Series(iris.target)

def visualize():
    s1 = features.iloc[0:50, :]
    s2 = features.iloc[50:100, :]
    s3 = features.iloc[100:150, :]

    plt.figure(figsize=(10, 6))
    plt.scatter(s1["sepal length (cm)"], s1["sepal width (cm)"], marker = '*', label = target_list[0])
    plt.scatter(s2["sepal length (cm)"], s2["sepal width (cm)"], marker = 'H', label = target_list[1])
    plt.scatter(s3["sepal length (cm)"], s3["sepal width (cm)"], marker = '+', label = target_list[2])
    plt.legend()
    plt.show()


Xtrain, Xtest, ytrain, ytest = train_test_split(features.values, target.values, test_size= 0.2)

def dtree():
    model = DecisionTreeClassifier()
    model.fit(Xtrain, ytrain)
    acc = model.score(Xtest, ytest)

    print(acc)
    # Decision Tree Accuracy : 90+ %

def svm():
    model = SVC()
    model.fit(Xtrain, ytrain)

    acc = model.score(Xtest, ytest)
    print(acc)
    # SVM Accuracy : 90-96%


svm()