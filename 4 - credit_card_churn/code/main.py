# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 17:34:38 2021

@author: kenhu
"""

import pandas as pd
import numpy as np
import mod_preparedata as prep
import mod4_models as mod
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import mod4_plotting_evaluation as pe
import xgboost as xgb
import seaborn as sns

df = prep.load_prep_data('BankChurners.csv')

df = prep.make_categorical_numeric(df)

X_train, X_test, y_train, y_test = prep.get_demographic_data(df)

X_train, X_test, y_train, y_test = prep.get_transaction_data(df)

X_train, X_test, y_train, y_test = prep.get_training_data(df)

y = df['Attrition_Status'].value_counts()

sns.barplot(x = y.index, y = y).set_title('Attrition Status, No Churn = 0, Churn = 1')
sns.title('Attrition Status, No churn = 0, Churn = 1')

dftable = df.iloc[:,4:9] #select all stocks 
def annret(x):
    result = (1+ x.mean())**52-1
    return result

def sd1y(x):
    result = x.std()*52**0.5
    return result

res = df.agg([np.mean,np.std,annret, sd1y])
#Correlation Matrix 
corrMatrix = df.corr()
print (corrMatrix)

plt.hist(df['Customer_Age'], bins  = 10,edgecolor='black', linewidth=1.2 )
plt.title('Customer Age')
plt.xlabel('Age')
plt.hist(df['Education_Level'], bins  = 8,edgecolor='black', linewidth=1.2 )


def boxplot(X): #create boxplot of feature distributions
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(10, 10))
    sns.boxplot(data=X, orient='h')
    plt.show()

def heat_map(df): #create heatmap of features
    corr = df.corr()

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(10, 10))
    g = sns.heatmap(corr, cmap='coolwarm', square=True, linecolor='w', linewidth=.5)
    plt.show()

heat_map(X_train)
boxplot(X_train)
X_train, X_test, y_train, y_test = prep.create_ND_data(df, ['Total_Trans_Amt', 'Total_Trans_Ct'])

X_train = prep.standardize_features(X_train)
X_test = prep.standardize_features(X_test)

all_models, all_names = mod.run_all_models(X_train, X_test, y_train, y_test)
key = pe.get_keymetrics(all_models, all_names, X_test, y_test)
pe.create_bar_plot(key, 'F1 score')
clf = mod.run_XGBoost_grid(X_train, X_test, y_train, y_test)
results = clf.cv_results_
best = clf.best_estimator_

best_score = clf.best_score_
best_params = clf.best_params_
key_best = pe.get_keymetrics([best], ['Best'], X_test, y_test)

best2 = xgb.XGBClassifier(learning_rate = 0.2, n_estimators = 300, reg_alpha = 0 , reg_lambda = 1)
best2.fit(X_train,y_train)
key_best2 = pe.get_keymetrics([best2], ['Best'], X_test, y_test)

best3_models, params = mod.run_XGB(X_train,X_test, y_train, y_test)


key3 = pe.get_keymetrics(best3_models, params, X_test, y_test)
best_model = best3.best_estimator_


ens_models = [best3_models[200],all_models[3],all_models[2], all_models[4]]
names = ['XGBoost','Random','Logistic','yes']
stacked = mod.run_Stacking(ens_models,names, X_train, y_train)
voting = mod.run_Voting(ens_models,names, X_train, y_train)

key_stack = pe.get_keymetrics([voting], 'Voting', X_test, y_test)

top_model = best3_models[78]
y_pred = top_model.predict(np.array(X_test))

cm = pe.create_confusion_matrix(y_test,y_pred)
disp = pe.graph_confusion_matrix(cm)
disp.plot()
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

model = RandomForestClassifier(n_estimators = 10000, random_state = 42, n_jobs = 6, verbose = 1)
model.fit(X_train,y_train)

prediction_test = model.predict(X_test)

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
ax = plt.subplot()
#ax.set_title("Input data")
# Plot the training points
ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, edgecolors="k")
# Plot the testing points
ax.scatter(
    X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
i += 

class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.arange(2), y = y_test)
