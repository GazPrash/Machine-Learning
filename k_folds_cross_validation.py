import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier


# StratifiedKFold is better than normal K-fold

dig = load_digits()
data = dig.data
target = dig.target

folds = StratifiedKFold(n_splits= 3)

for train_index, test_index in folds.split(data, target):
    Xtrain, Xtest, ytrain, ytest = data[train_index], data[test_index], \
                                   target[train_index], target[test_index]

        # append different scores in a list by using .score() of that estimator
        

# cross_val_score does the same thing done in the above loop in an abstracted manner.
print(cross_val_score(RandomForestClassifier(), data, target))