import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


dig = load_digits()
data = dig.data
target = dig.target

xtrain, xtest, ytrain, ytest = train_test_split(data, target, test_size= 0.2)
model1 = DecisionTreeClassifier()
model1.fit(xtrain, ytrain)
acc1 = model1.score(xtest, ytest)


model2 = SVC(kernel='rbf')
model2.fit(xtrain, ytrain)
acc2 = model2.score(xtest, ytest)

# PREDICTING
def predict_number(
    model_type:str,
    img_data,
    matrix_data
):
    if model_type == "DecisionTree":
        output = model1.predict([matrix_data])
        print(f"The predicted number is : {output}")    
        print(f"(Model Accuracy) : {acc1}")    

    elif model_type == "SVM":
        output = model2.predict([matrix_data])
        print(f"The predicted number is : {output}")    
        print(f"(Model Accuracy) : {acc2}")    

    plt.imshow(img_data, interpolation='nearest')
    plt.show()

predict_number("SVM", dig.images[455], dig.data[455])
