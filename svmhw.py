# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 23:06:58 2018

@author: arfas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features.
y = iris.target






from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.25,random_state=0)




from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test =sc.transform(X_test)



# fitting the classifier to the training set 

classifier=SVC(kernel='linear')


classifier.fit(X_train,y_train)

# predicting the test set results
y_pred =classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


# Visualizing the training test results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train

x1 = np.arange(X_set[:, 0].min(), X_set[:, 0].max(), step = 0.01)
x2 = np.arange(X_set[:, 1].min(), X_set[:, 1].max(), step = 0.01)
X1, X2 = np.meshgrid(x1,x2)


Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)

plt.contourf(X1, X2, Z.reshape(X1.shape), alpha = 0.55, 
             cmap = ListedColormap(('red', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for j in np.unique(y_set):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'blue'))(j), label = j)
    
plt.title('KNN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated salary')
plt.legend()
plt.show()


# Visualizing the training test results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test

x1 = np.arange(X_set[:, 0].min(), X_set[:, 0].max(), step = 0.01)
x2 = np.arange(X_set[:, 1].min(), X_set[:, 1].max(), step = 0.01)
X1, X2 = np.meshgrid(x1,x2)


Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)

plt.contourf(X1, X2, Z.reshape(X1.shape), alpha = 0.55, 
             cmap = ListedColormap(('red', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for j in np.unique(y_set):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'blue'))(j), label = j)
    
plt.title('KNN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated salary')
plt.legend()
plt.show()