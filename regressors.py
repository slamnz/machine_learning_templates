# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 13:28:42 2017

@author: Sunny Lam
"""
# Example Dataset 

from sklearn.datasets import load_boston
data = load_boston()

X = data.data
y = data.target

#

from sklearn.model_selection import cross_val_score

# Metrics

def test_results(model, X, y):

    try:
        from sklearn.model_selection import cross_val_score
        compute = cross_val_score(model, X, y, cv=10)
        mean = compute.mean()
        std = compute.std()
        print(type(model).__name__)
        print("Average Score: %.2f (+/- %.2f) \n" % (abs(mean),std))
    except Exception as e:
        print(e)
        print("Failed.")
# Support Vector Machines

from sklearn.svm import SVR, LinearSVR, NuSVR

regressor = SVR()
test_results(regressor, X, y)

regressor = LinearSVR()
test_results(regressor, X, y)

regressor = NuSVR()
test_results(regressor, X, y)

from sklearn.kernel_ridge import KernelRidge

regressor = KernelRidge()
test_results(regressor, X, y)

from sklearn.linear_model import SGDRegressor

regressor = SGDRegressor()
test_results(regressor, X, y)

# KNeighbors

from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

regressor = KNeighborsRegressor()
test_results(regressor, X, y)

#regressor = RadiusNeighborsRegressor()
#test_results(regressor, X, y)

#Gaussian Process

from sklearn.gaussian_process import GaussianProcessRegressor

regressor = GaussianProcessRegressor()
test_results(regressor, X, y)

#Decision Trees

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()
test_results(regressor, X, y)

from sklearn.ensemble import GradientBoostingRegressor

regressor = GradientBoostingRegressor()
test_results(regressor, X, y)

from sklearn.isotonic import IsotonicRegression

#regressor = IsotonicRegression()
#test_results(regressor, X, y)

from sklearn.neural_network import MLPRegressor

regressor = MLPRegressor()
test_results(regressor, X, y)