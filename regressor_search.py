# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 21:56:53 2017

@author: S
"""

def get_results(model, X, y):

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sklearn.model_selection import cross_val_score
        compute = cross_val_score(model, X, y)
        mean = compute.mean()
        std = compute.std()
        return mean, std

def display_regressor_results(X,y):

    models = []
    
    from sklearn.tree import DecisionTreeRegressor

    models += [DecisionTreeRegressor()]

    from sklearn.ensemble import GradientBoostingRegressor

    models += [GradientBoostingRegressor()]
    
    from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

    models += [KNeighborsRegressor(), RadiusNeighborsRegressor()]

    from sklearn.neural_network import MLPRegressor
    models += [MLPRegressor(hidden_layer_sizes=(len(X.columns), 2))]
    
    from xgboost import XGBRegressor
    models += [XGBRegressor()]

    output = {}

    for m in models:
        try:
            model_name = type(m).__name__
            scores = get_results(m,X,y)
            row = {"Average Score" : scores[0], "Standard Deviation" : scores[1]}
            output[model_name] = row
        except:
            pass

    from pandas import DataFrame
    from IPython.display import display

    display(DataFrame(data=output).T.round(2).sort_values("Average Score", ascending=False))

display_regressor_results(X,y)