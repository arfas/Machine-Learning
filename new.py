# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:30:40 2018

@author: arfas
"""

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

 

data = pd.read_csv('Social_Network_Ads.csv')

 

data1 = pd.get_dummies(data, columns = ['Gender'])

 

X = data.iloc[:, [2,3]]

Y = data.iloc[:, 4]

 

#importing train test split method

from sklearn.cross_validation import train_test_split

 

#splitting the data set into training and test data sets

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.25, random_state = 0)

 

 

#Feature Scaling

from sklearn.preprocessing import StandardScaler

 

sc = StandardScaler()

 

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

 

#Fitting classifier to the training set

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 8, metric = 'minkowski', p=2)

 

#Create your classifier here

classifier.fit(X_train,Y_train)

 

#Predicting the test set results

Y_pred = classifier.predict(X_test)

 

 

#Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm =confusion_matrix(Y_test, Y_pred)

 

#Visualizing the Training set results

from matplotlib.colors import ListedColormap

X_set, Y_set = X_train, Y_train

 

X1 = np.arange(X_set[:, 0].min(),X_set[:, 0].max(),step = 0.1)

X2 = np.arange(X_set[:, 1].min(),X_set[:, 1].max(),step = 0.1)

 

X1,X2 = np.meshgrid(X1,X2)

 

Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)

 

plt.contourf(X1, X2, Z.reshape(X1.shape), alpha = 0.55,

             cmap = ListedColormap(('red', 'blue')))

 

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

 

for j in np.unique(Y_set):

    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j,1],

                c = ListedColormap(('red', 'blue'))(j),label = j)

   

    plt.title('KNN (Training set)')

    plt.xlabel('Age')

    plt.ylabel('Estimated Salary')

    plt.legend()

    plt.show()
    

   