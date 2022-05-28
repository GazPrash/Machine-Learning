# Exercise: Machine Learning Finding Optimal Model and Hyperparameters
# For digits dataset in sklearn.dataset, please try following classifiers
# and find out the one that gives best performance.
 
# Also find the optimal parameters for that classifier.

import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV


digs = load_digits()
data = digs.data
target = digs.target

model_grid = {
    "rforest" : {
        "model" : RandomForestClassifier(),
        "params" : {
            'n_estimators' : [5, 10, 20]
        }
    },
    "svm" : {
        "model" : svm.SVC(),
        "params" : {
            "C" : [1, 5, 10],
            "kernel" : ['linear', "rbf"]
        }
    },
    "linear" : {
        "model" : LogisticRegression(max_iter=1e5),
        "params" : {}
    },
    "gaussian_bayes" : {
        "model" : GaussianNB(),
        "params" : {}
    },
    "multinomial_bayes" : {
        "model" : MultinomialNB(),
        "params" : {}
    },
    "decision_tree" : {
        "model" : DecisionTreeClassifier(),
        "params"  : {
            "criterion" : ["gini", "entropy"]
        }
    }

}

results = []
for model_name, model_atts in model_grid.items():
    clf = GridSearchCV(model_atts["model"], model_atts["params"], cv=5, return_train_score= False)
    clf.fit(data, target)
    results.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })


results = pd.DataFrame(results)
print(results)
# Hence we find out that the best model for this dataset is SVC, with a 97.3% Accuracy
# with C value of 5 & Rbf Kernel, Followed up by Random Forest & Logistic
