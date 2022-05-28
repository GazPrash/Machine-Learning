from pyexpat import model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, BernoulliNB

import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv(
    "https://raw.githubusercontent.com/codebasics/py/master/ML/9_decision_tree/Exercise/titanic.csv"
)
# Exploring & Cleaning Data | Declaring Features & Target
data["Age"] = data["Age"].fillna(data.Age.median())

features = data[["Pclass", "Age", "Fare"]]

lencoder = LabelEncoder()
features_discrete = pd.DataFrame()
features_discrete["Sex"] = data["Sex"]
features_discrete["Sex"] = lencoder.fit_transform(features_discrete["Sex"])

target = data["Survived"]


xtrain, xtest, ytrain, ytest = train_test_split(features.values, target.values, test_size= 0.2)
disc_xtrain, disc_xtest, disc_ytrain, disc_ytest = train_test_split(
                                                    features_discrete[["Sex"]].values, 
                                                    target.values
                                                )


model1 = GaussianNB()
model2 = BernoulliNB()

model1.fit(xtrain, ytrain)
model2.fit(disc_xtrain, disc_ytrain)

tc_number = 32
prob1 = model1.predict_proba([xtest[tc_number]])
probabiliy1 = (prob1[0][0])

prob2 = model2.predict_proba([disc_xtest[tc_number]])
probabiliy2 = (prob2[0][0])

print(f"Final Probability : {round(probabiliy1 * probabiliy2 * 100, 2)}")
print(xtest[tc_number])
