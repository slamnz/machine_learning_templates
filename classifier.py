from sklearn.datasets import load_iris
import sklearn as sk

data = load_iris()

X = data.data
y = data.target

# Test Method

def test_results(model, X, y):

    compute = sk.model_selection.cross_val_score(model, X, y, cv=10)
    mean = compute.mean()
    std = compute.std()
    print(type(model).__name__)
    print("Average Score: %.2f (+/- %.2f) \n" % (mean,std))

# Decision-Tree

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
test_results(classifier, X, y)

# Naive Bayes

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

classifier = GaussianNB()
test_results(classifier, X, y)

classifier = MultinomialNB()
test_results(classifier, X, y)

classifier = BernoulliNB()
test_results(classifier, X, y)

# Ensemble Classifiers

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10)
test_results(classifier, X, y)

from sklearn.ensemble import AdaBoostClassifier

classifier = AdaBoostClassifier(n_estimators=100)
test_results(classifier, X, y)

from sklearn.ensemble import GradientBoostingClassifier

classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0)
test_results(classifier, X, y)

from sklearn.ensemble import ExtraTreesClassifier

classifier = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
test_results(classifier, X, y)

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
classifier = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
test_results(classifier, X, y)

# Linear Discriminants

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
classifier = LinearDiscriminantAnalysis()
test_results(classifier, X, y)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
classifier = QuadraticDiscriminantAnalysis()
test_results(classifier, X, y)

# Neural Networks

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 2), random_state=1)
test_results(classifier, X, y)

# SVM

from sklearn.svm import SVC, LinearSVC

classifier = SVC(decision_function_shape='ovo')
test_results(classifier, X, y)

classifier = LinearSVC()
test_results(classifier, X, y)

from sklearn.linear_model import SGDClassifier

classifier = SGDClassifier()
test_results(classifier, X, y)

from sklearn.neighbors.nearest_centroid import NearestCentroid
classifier = NearestCentroid()
test_results(classifier, X, y)