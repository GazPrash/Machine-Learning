# K Nearest Neighbour Algorithm can help us classify a point into different available clusters 

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()
data = iris.data
target = iris.target

xtrain, xtest, ytrain, ytest = train_test_split(data, target)

# Building KNN Classifier

para_grid = {
    "model" : KNeighborsClassifier(),
    "params" : {
        "n_neighbors" : [3, 5, 7]
    }
}

scores = []
model = GridSearchCV(para_grid["model"], para_grid["params"], cv=5, return_train_score=False)
model.fit(xtrain, ytrain)
scores.append({
    'model': "KNN Classifier",
    'best_score': model.best_score_,
    'best_params': model.best_params_
})
    
df = pd.DataFrame(scores, columns=['model','best_score','best_params'])
print(df)
