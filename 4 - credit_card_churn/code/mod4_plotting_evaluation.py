# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 15:34:08 2021

@author: kenhu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import PrecisionRecallDisplay
from matplotlib import cm
import matplotlib
#%matplotlib inline
from IPython.display import display

def precision_recall_graph(y_test, prediction_test): #create Precision-Recall ROC curve
    display = PrecisionRecallDisplay.from_predictions(
        y_test, prediction_test, name="LinearSVC"
    )
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    return display


def create_confusion_matrix(y_test, y_pred): #create confusion matrix
    y_test = np.array(y_test)
    
    cm = metrics.confusion_matrix(y_test,y_pred)
    metrics.ConfusionMatrixDisplay(cm)
    return cm

def graph_confusion_matrix(cm): #plot confusion matrix
    disp = metrics.ConfusionMatrixDisplay(cm)
    return disp


def get_feature_importance(X, model): #feature importances for ensembling methods
    feature_list = list(X.columns)
    feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
    
    feature_importance = pd.DataFrame(columns = ['Importances'], index = feature_list)
    feature_importance['Importances'] = model.feature_importances_
    feature_importance = feature_importance['Importances'].sort_values(ascending=False)
    
    return feature_importance

def get_keymetrics(models_list, names, X_test, y_test): #key metrics from predicted values. returns a dataframe with key metrics for each model
    jac_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for model in models_list:
        y_pred = model.predict(X_test)
        jac_scores.append(metrics.jaccard_score(y_test, y_pred))
        accuracy_scores.append(metrics.accuracy_score(y_test,y_pred))
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(y_test, y_pred)
        precision_scores.append(precision[1])
        recall_scores.append(recall[1])
        f1_scores.append(fscore[1])

    
    df = pd.DataFrame({'names': names, 'Jaccard Scores' : jac_scores, 'Accuracy' : accuracy_scores, 'Precision' : precision_scores,
                       'Recall' : recall_scores, 'F1 score' : f1_scores})
    return df

def create_bar_plot(df, metric = 'F1 score'):   #bar plot for comparing metrics, default is F1 score
    g = sns.barplot(x = df[metric],y = df['names'],data = df[metric])
    g.set_xlabel(metric)
    g = g.set_title(metric + ' scores')



def two_D_mesh(model, X_train, X_test, y_train, y_test, h1 = 1, h2 = 1):   #For 2-Dimensional classification visualization
    x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
    y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h1),
                         np.arange(y_min, y_max, h2))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    fig = plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap = 'bwr', alpha = 0.5)
    
    # Plot also the training points
    plt.scatter(x=X_train.iloc[:, 0], y=X_train.iloc[:, 1], c = y_train ,cmap='bwr', alpha=0.5, edgecolor="black")
    plt.scatter(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], c = y_test ,cmap='bwr', alpha=1.0, edgecolor="black")

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    return fig

def multi_2D_mesh(model_list, name_list, X_train, X_test, y_train, y_test, h1 = 1, h2 = 1): #create multi 2D plots stacked up
    figure = plt.figure(figsize=(60, 9))
    i = 0
    for name, model in zip(name_list, model_list):
        x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
        y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h1),
                      np.arange(y_min, y_max, h2))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax = plt.subplot(1, len(model_list), i + 1)
        ax.scatter(x=X_train.iloc[:, 0], y=X_train.iloc[:, 1], c = y_train ,cmap='bwr', alpha=0.5, edgecolor="black")
        ax.scatter(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], c = y_test ,cmap='bwr', alpha=1.0, edgecolor="black")

        ax.contourf(xx, yy, Z, cmap = 'bwr', alpha = 0.5)
        ax.set_title(name)
        i += 1
    plt.show()

def heat_map(df): #create heatmap of features
    corr = df.corr()

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(10, 10))
    g = sns.heatmap(corr, cmap='coolwarm', square=True, linecolor='w', linewidth=.5)
    plt.show()

def boxplot(X): #create boxplot of feature distributions
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(10, 10))
    sns.boxplot(data=X, orient='h')
    plt.show()

