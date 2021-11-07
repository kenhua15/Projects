# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 15:10:17 2021

@author: kenhu
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import expit

df = pd.read_csv('BankChurners.csv')
df = df.iloc[: , :-2]
df['Attrition_Status'], index = pd.factorize(df['Attrition_Flag'])

df_plot = df
sns.pairplot(df, hue='Attrition_Flag')

##Logistic

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import neighbors, datasets


X = df['Total_Trans_Ct']
y = df['Attrition_Status']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

X_train = np.array(X_train).reshape(-1,1)
y_train = np.array(y_train)
X_test = np.array(X_test).reshape(-1,1)
y_test = np.array(y_test)

lml = LogisticRegression()
lml.fit(X_train, y_train)


y_pred = lml.predict(X_test)
score = lml.score(X_test,y_test)

plt.figure(1, figsize=(4, 3))
plt.scatter(X.ravel(), y, color='black', zorder=20)
X_test = np.linspace(0, 130, 300)
loss = expit(X_test * lml.coef_ + lml.intercept_).ravel()
plt.plot(X_test, loss, color='red', linewidth=3)


from sklearn.neighbors import KNeighborsClassifier

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

X = df.select_dtypes(include=numerics)
X = X.drop(columns = ['Attrition_Status'])

X = df[['Months_on_book', 'Credit_Limit']]

y = df['Attrition_Status']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
 
neigh = KNeighborsClassifier(n_neighbors = 5)
neigh.fit(X_train, y_train)

score = neigh.score(X_test, y_test)

clf = neighbors.KNeighborsClassifier(n_neighbors = 5)
clf.fit(X, y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
h1 = 0.2
h2 = 20

x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h1),
                     np.arange(y_min, y_max, h2))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z)

# Plot also the training points
sns.scatterplot(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], hue = y_test , alpha=1.0, edgecolor="black")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

from sklearn.ensemble import RandomForestClassifier

