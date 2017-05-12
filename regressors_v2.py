# -*- coding: utf-8 -*-
"""
Created on Fri May 12 21:45:09 2017

@author: Sunny Lam
"""

from numpy import log1p
def squared_logarithmic_error(y_true, y_pred):
    return (log1p(y_pred) - log1p(y_true)) ** 2
def mean_squared_logarithmic_error(y_true, y_pred):
    calculation = squared_logarithmic_error(y_true, y_pred)
    return calculation.sum() / len(calculation)
def root_mean_squared_logarithmic_error(y_true, y_pred):
    return mean_squared_logarithmic_error(y_true, y_pred) ** 0.5

# === === #

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

def score_row(actuals,predictions):

    parameters = {"y_true" : actuals,
                 "y_pred" : predictions}

    score_functions = [explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, root_mean_squared_logarithmic_error]

    output = {}

    for func in score_functions:
        output[str(func.__name__)] = func(**parameters) 

    return output

# === === #

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

def score_row(actuals,predictions):

    parameters = {"y_true" : actuals,
                 "y_pred" : predictions}

    #score_functions = [explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, root_mean_squared_logarithmic_error]
    score_functions = [r2_score, explained_variance_score, mean_absolute_error, root_mean_squared_logarithmic_error]
    
    output = {}

    for func in score_functions:
        output[str(func.__name__)] = func(**parameters) 

    return output

from sklearn.model_selection import KFold

def cross_val_score(model, data, features, target_feature):

    iterations = []

    splits = 10
    splitter = KFold(n_splits=splits, random_state=0)
    i = iter(range(0,splits))
    score_rows = []

    for train, test in splitter.split(data):

        training_set = data.iloc[train]
        testing_set = data.iloc[test]

        model.fit(training_set[features],training_set[target_feature])
        iterations += [model]

        predictions = model.predict(testing_set[features])
        actuals = testing_set[target_feature]

        # === Score Metrics ===

        score_rows += [score_row(actuals,predictions)]
        
    return score_rows

from IPython.display import display
from pandas import DataFrame

def display_mean_scores(model, data, features, target):
    print(type(model).__name__)
    display(DataFrame(cross_val_score(model,data,features,target)).mean())
    
from pandas import options
def display_cv_scores(model, data, features, target):
    options.display.float_format = '{:,.3f}'.format
    display(DataFrame(cross_val_score(model,data,features,target)).round(2))

from time import time
from pandas import Series
    
def regressor_runthrough(regressors, data, features, target_feature):
    results = {}
    for r in regressors:
        key = type(r).__name__
        try:
            start = time()
            
            unit = DataFrame(cross_val_score(r,data,features,target_feature)).mean()
            
            finished = time() - start
            
            unit = unit.append(Series([finished], index=["Total Processing Time"]))
            
            results[key] = unit
            
        except:
            pass
            #print(key + " failed.")
    return DataFrame(results).T

# === === #

regressors = []

from xgboost import XGBRegressor
regressors += [XGBRegressor()]

from sklearn.linear_model import HuberRegressor, PassiveAggressiveRegressor, RANSACRegressor, SGDRegressor,TheilSenRegressor
regressors += [HuberRegressor(), PassiveAggressiveRegressor(), RANSACRegressor()]

from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
regressor = RadiusNeighborsRegressor()
regressors.append(regressor)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressors.append(regressor)

from sklearn.tree import ExtraTreeRegressor
regressor = ExtraTreeRegressor()
regressors.append(regressor)

from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
regressor = AdaBoostRegressor()
regressors.append(regressor)
regressor = BaggingRegressor()
regressors.append(regressor)
regressor = ExtraTreesRegressor()
regressors.append(regressor)
regressor = RandomForestRegressor()
regressors.append(regressor)
regressor = GradientBoostingRegressor()
regressors.append(regressor)

