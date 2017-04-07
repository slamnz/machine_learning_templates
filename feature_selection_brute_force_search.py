# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:11:38 2017

@author: Sunny Lam
"""

# 1. Instantiate Dataset

from pandas import read_csv
data = read_csv("<FILEPATH HERE>")
X = "<FEATURE DATA>"
y = "<TARGET DATA>"

# 2. Instantiate a Model

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(learning_rate=0.1, max_depth=10, max_features=1.0, min_samples_leaf=1, min_samples_split=8, subsample=0.75)

# 3. The Feature Brute-Search Function

from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from itertools import combinations

def combination_score_dataframe(model, X, y):

    features = X.columns
    current = {}
    
    for i in range(3, len(X)+1):
        for j in combinations(features, i):
            scores = cross_val_score(model,X[list(j)],y,cv=5)
            unit = {}
            unit["Selection Count"] = "%s/%s" % (len(j),len(features))
            unit["Average Score"] = scores.mean()
            unit["Standard Deviation"] = scores.std()
            current[str(list(j))] = unit
    
    output = DataFrame(current).T
    
    return output

# 4. Run experiment

result = combination_score_dataframe(model,X,y)

# 4. Results as DataFrame

result
