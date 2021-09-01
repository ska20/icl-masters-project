import numpy as np
from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier

from joblib import Parallel, delayed

def train_model(X_train, y_train, X_test, y_test, model_name, seed):
    if model_name == 'logisticregression':
        model = LogisticRegression()
    elif model_name == 'randomforest':
        model = RandomForestClassifier(max_depth=2)
    elif model_name == 'gaussiannb':
        model = GaussianNB()
    elif model_name == 'bernoullinb':
        model = BernoulliNB()
    elif model_name == 'svmrbf':
        model = svm.SVC()
    elif model_name == 'gbm':
        model = GradientBoostingClassifier(max_depth=2)
    elif model_name == 'extratrees':
        model = ExtraTreesClassifier(max_depth=2)
    elif model_name == 'lda':
        model = LinearDiscriminantAnalysis()
    elif model_name == 'passiveagressive':
        model = PassiveAggressiveClassifier()
    elif model_name == 'adaboost':
        model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2))
    elif model_name == 'bagging':
        model = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=2))

    if model_name=='passiveagressive':
        model.fit(X_train, y_train) # Fit training data
        predict = model.decision_function(X_test)
    elif model_name=='svmrbf':
        model.fit(X_train, y_train) # Fit training data
        predict = model.predict(X_test)
    else:
        model.fit(X_train, y_train) # Fit training data
        predict = model.predict_proba(X_test)[:,1]

    # AUC Computation
    auc = metrics.roc_auc_score(y_test, predict)
    return auc

def parallel_train(X_train, y_train, X_test, y_test, model, nc):
    # Run jobs in parallel for heightened efficiency
    result = Parallel(n_jobs=-1)(delayed(train_model)(X_train, y_train, X_test, y_test, model, seed) for seed in range(nc))
    return max(result)
