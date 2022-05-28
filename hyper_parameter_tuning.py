# Selecting a model or optimal hyper-parameters for that model for training
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV


iris = load_iris()
data = iris.data
target = iris.target


clf = GridSearchCV(
    SVC(gamma="auto"),
    {
        'C' : [1, 10, 20],
        'kernel' : ['linear', 'rbf'],

    },
    cv=5,
    return_train_score=False
)

clf.fit(data, target)
cvdf = pd.DataFrame(clf.cv_results_)

# print(cvdf[["param_C", "param_kernel", "mean_test_score"]])

# Using GridSearchCV can get a little exaggerated when working with larger datasets, because
# it will try to permute all the possible results, hence we try to use Rand. SearchCV which 
# has a more practical approach and can allow us to limit the no. of iterations as well.

clf2 = RandomizedSearchCV(
    SVC(gamma="auto"),
    {
        'C' : [1, 10, 20],
        'kernel' : ['linear', 'rbf'],

    },
    cv=5,
    return_train_score=False,
    n_iter=2

)


# Choosing the best fit model

model_para_grid = {
    "rforest" : {
        "model" : RandomForestClassifier(),
        "params" : {
            "n_estimators" : [1, 5, 10]
        }
    },

    "svm" : {
        "model" : SVC(),
        "params" : {
            "C" : [1, 10, 20],
            "kernel" : ["linear", "rbf"]
        }
    },

    "logistic" : {
        "model" : LogisticRegression(),
        "params" : {
            "C" : [1, 5, 10],
            "max_iter" : [1e5]
        }
    },

    "linear" : {
        "model" : LinearRegression(),
        "params" : {
        }
    }
}


scores = []

for model_name, mp in model_para_grid.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df)