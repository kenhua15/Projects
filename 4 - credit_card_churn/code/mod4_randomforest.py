# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 14:57:38 2021

@author: kenhu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('BankChurners.csv')
df = df.iloc[: , :-2]
df['Attrition_Status'], index = pd.factorize(df['Attrition_Flag'])

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
X = df.select_dtypes(include=numerics)
X = X.drop(columns = ['Attrition_Status'])
y = df['Attrition_Status']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

model = RandomForestClassifier(n_estimators = 10000, random_state = 42, n_jobs = 6, verbose = 1)
model.fit(X_train,y_train)

prediction_test = model.predict(X_test)

print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))

###Feature importance

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)

ff = pd.DataFrame(columns = ['Importances'], index = feature_list)
ff['Importances'] = model.feature_importances_
ff= ff['Importances'].sort_values(ascending=False)


### Confusion Matrix

from sklearn.metrics import confusion_matrix, accuracy_score #sreeni

def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.show()


accuracy = accuracy_score(np.array(y_test), prediction_test)
print ("accuracy = ", accuracy)
cm = confusion_matrix(np.argmax(y_test,axis=0), np.argmax(prediction_test,axis=0))
print (cm)

showconfusionmatrix(cm)
