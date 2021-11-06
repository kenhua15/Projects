# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 14:33:41 2021

@author: kenhu
"""

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

def run_all_models(X_train, X_test, y_train, y_test):   #
    
    classifiers = [
        KNeighborsClassifier(n_neighbors = 3),
        LogisticRegression(solver='lbfgs', max_iter=1000),
        DecisionTreeClassifier(max_depth = 7, class_weight = 'balanced'),
        RandomForestClassifier(n_estimators = 100, random_state = 42, n_jobs = 6, verbose = 1),
        AdaBoostClassifier(),
        GaussianNB(),
        xgb.XGBClassifier()
        ]
    
    names = [
        'KNeighborsClassifier', 
        'Logistic Regression', 
        'Decision Tree Classifier', 
        'Random Forest Classifier',
        'Ada Boost',
        'Naive Bayes',
        'XG Boost'
        ]
    
    for model in classifiers:
        model.fit(X_train, y_train)
        
    return classifiers, names

def run_RandomForest(X_train, X_test, y_train, y_test): #run Grid Search with Train dataset for hyperparameter optimization for Random Forest
    
    n_estimators = [100, 1000]
    min_samples_split = [2,4,6]
    min_samples_leaf = [1,2,3]
    max_depth = [3,5, None]
    
    model_list = []
    model_params = []
    combined = [(i,j,k,l) for i in n_estimators for j in min_samples_split for k in min_samples_leaf for l in max_depth]

    for params in combined:
        model = RandomForestClassifier(n_estimators = params[0],min_samples_split = params[1],
                                       min_samples_leaf = params[2], max_depth = params[3], random_state = 42, n_jobs = 6, verbose = 1)
        model.fit(X_train,y_train)
        model_list.append(model)
        # model_param_temp =
        # model_params = 
    
    return model_list, combined

def run_RandomForest_grid(X_train, X_test, y_train, y_test):  #run Grid Search with cross validation for hyperparameter optimization for Random Forest
    parameters = {'min_samples_split':[2,4,6], 'n_estimators':[100, 1000]}
    model = RandomForestClassifier()
    clf = GridSearchCV(model, parameters)
    clf.fit(X_train, y_train)
    
    
    return clf

def run_XGBoost_grid(X_train, X_test, y_train, y_test): #run Grid Search with cross validation for hyperparameter optimization for XGBoost
    param_grid = {'gamma': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4, 200],
              'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.300000012, 0.4, 0.5, 0.6, 0.7],
              'max_depth': [5,6,7,8,9,10,11,12,13,14],
              'n_estimators': [50,65,80,100,115,130,150],
              'reg_alpha': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200],
              'reg_lambda': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200]}
    parameters = {'reg_lambda': [1],'reg_alpha': [0],'learning_rate':[0.2,0.3,0.7], 'n_estimators':[50,100,300]}

    model = xgb.XGBClassifier()
    clf = GridSearchCV(model, parameters)
    clf.fit(X_train, y_train)
    
    
    return clf

def run_XGB(X_train, X_test, y_train, y_test): #run Grid Search with Train dataset for hyperparameter optimization for XGBoost
    
    n_estimators = [ 50, 150, 100,200,300]
    learning_rates = [0.2,0.3,0.25,0.7,0.6,0.45]
    reg_alpha = [0,0.5,0.3, 0.7]
    scale_pos_weight  = [0.85,0.7,1, 0.5]
    
    model_list = []
    model_params = []
    combined = [(i,j,k,l) for i in n_estimators for j in learning_rates for k in reg_alpha for l in scale_pos_weight ]

    for params in combined:
        model = xgb.XGBClassifier(n_estimators = params[0],learning_rate = params[1],
                                       reg_alpha = params[2], scale_pos_weight  = params[3], random_state = 42, n_jobs = 7, verbose = 1)
        model.fit(X_train,y_train)
        model_list.append(model)
        # model_param_temp =
        # model_params = 
    
    return model_list, combined

def run_Voting(model_list, names, X_train, y_train, voting = 'hard', weights = []): #ensembling method with voting
    model_list = list(zip(names,model_list))
    voting_classifier = VotingClassifier(estimators=model_list,
                                    voting='hard', #<-- sklearn calls this hard voting
                                    n_jobs=-1)
    voting_classifier.fit(X_train, y_train)
    return voting_classifier

def run_Stacking(model_list, names, X_train, y_train): #ensembling method with stacking
    model_list = list(zip(names,model_list))
    stacked = StackingClassifier(estimators=model_list, final_estimator=LogisticRegression())    
    stacked.fit(X_train, y_train)
    return stacked


# mods, model_params = run_RandomForest(X_train, X_test, y_train, y_test)                    
# clf = run_RandomForest_grid(X_train, X_test, y_train, y_test) 

# clf.get_params()
# votes = run_Voting(all_models, all_names, X_train, y_train)
# stacked = run_Stacking(all_models, all_names, X_train, y_train)
