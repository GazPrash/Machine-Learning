import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC

# Download heart disease dataset heart.csv in Exercise folder and do following, 
# (credits of dataset: https://www.kaggle.com/fedesoriano/heart-failure-prediction)

# * Load heart disease dataset in pandas dataframe
# * Remove outliers using Z score. Usual guideline is to remove anything that has Z score > 3 
#   formula or Z score < -3
# * Convert text columns to numbers using label encoding and one hot encoding
# * Apply scaling
# * Build a classification model using various methods (SVM, logistic regression, random forest) 
#   and check which model gives you the best accuracy
# * Now use PCA to reduce dimensions, retrain your model and see what impact it has on your model 
#   in terms of accuracy. Keep in mind that many times doing PCA reduces the accuracy but 
#   computation is much lighter and that's the trade off you need to consider while building 
#   models in real life

data = pd.read_csv(
    "https://raw.githubusercontent.com/codebasics/py/master/ML/18_PCA/Exercise/heart.csv"
)

# Utility Funcs
def adjust_df_zscore_index(df, para:str):
    return (df[para].mean(), df[para].std())

def predict_heart_disease(model:RandomForestClassifier, test_case:list):
    return model.predict([test_case])


neu, stdev = adjust_df_zscore_index(data, "Cholesterol")
remove_df1 = data[((data["Cholesterol"] - neu)/stdev > 3)]
data.drop(remove_df1.index, axis = 0)

neu, stdev = adjust_df_zscore_index(data, "MaxHR")
remove_df2 = data[((data["MaxHR"] - neu)/stdev > 3)]
data.drop(remove_df2.index, axis = 0)


data["Sex"] = data["Sex"].apply(lambda x : 1 if x == "M" else 0)
data["ExerciseAngina"] = data["ExerciseAngina"].apply(lambda x : 1 if x == "Y" else 0)

features = pd.DataFrame()
features = pd.get_dummies(data[["ChestPainType", "RestingECG", "ST_Slope"]], drop_first=1)

target = data["HeartDisease"]

features = pd.concat([
    features, 
    data[[
        "Age", 
        "Sex", 
        "RestingBP", 
        "Cholesterol",
        "FastingBS",
        "MaxHR",
        "ExerciseAngina",
        "Oldpeak"
        ]]],
        axis= 1
    )

# print(features)

xtrain, xtest, ytrain, ytest = train_test_split(features.values, target.values)

param_grid = {
    "rforest" : {
        "model" : RandomForestClassifier(),
        "params" : {
            "n_estimators" : [10, 25, 50, 100],
            "max_depth" : [4, 5, 10]
        }
    },
    
    "svm" : {
        "model" : SVC(),
        "params" : {
            "C" : [1, 3, 5, 10],
        }
    },

    "naive_bayes" : {
        "model" : BernoulliNB(),
        "params" : {}
    }

}

scores = []
for model_name, model_props in param_grid.items():
    clf_model = GridSearchCV(model_props["model"], model_props["params"], cv=5, return_train_score=0)
    clf_model.fit(xtrain, ytrain)
    scores.append({
        'Model': model_name,
        'Highest Score': clf_model.best_score_,
        'Recommended Parameters': clf_model.best_params_
    })
    print(f"Training for {model_name} ... Completed.")

perf = pd.DataFrame(scores, columns=['Model','Highest Score','Recommended Parameters'])
# print(perf)

# Since RFC is the most accurate we will train it
model = RandomForestClassifier(n_estimators=100)
model.fit(xtrain, ytrain)

pred1 = predict_heart_disease(model, xtest[0])

print(pred1)

