import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_wine

# Machine Learning Tutorial - Naive Bayes: Exercise
# Use wine dataset from sklearn.datasets to classify wines into 3 categories.
#  
# Load the dataset and split it into test and train. After that train the model 
# using Gaussian and Multinominal classifier and post which model performs better. 
# Use the trained model to perform some predictions on test data.

wine = load_wine()
data = wine.data
target = wine.target

xtrain, xtest, ytrain, ytest = train_test_split(data, target)

model1 = GaussianNB()
model1.fit(xtrain, ytrain)
acc1 = model1.score(xtest, ytest)
print(f"Guassiab NB Accuracy : {acc1}")

model2 = MultinomialNB()
model2.fit(xtrain, ytrain)
acc2 = model2.score(xtest, ytest)
print(f"Multinomial NB Accuracy : {acc2}")

# OUTPUT : 
# Guassiab NB Accuracy : 0.9777777777777777
# Multinomial NB Accuracy : 0.8888888888888888

