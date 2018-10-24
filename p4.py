# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 22:33:47 2018

@author: arfas
"""
#Name:Mohammed Arfa Adeeb
#ID:999993905
#Description:Import the data set from sklearn library,Split the data set into training set and test set,Fitting K-Nearest Neighbors to the Training set,Predicting the Test set results,Making the Confusion Matrix,Visualizing the Training set results
#Visualizing the Testing set results, Fitting Linear SVM to the Training set.
#1)KNN


import numpy as np 
import matplotlib.pyplot as plt 

from sklearn import datasets 

iris = datasets.load_iris() 

X = iris.data[:, :2] 
y = iris.target 

#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#KNN 

#Creating Classifier
from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors=6, metric = 'minkowski', p=2); 

#Fitting Classifier to the Training set
classifier.fit(X_train, y_train)

#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Visualizing the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train

x1 = np.arange(X_set[:,0].min(), X_set[:,0].max(), step=0.01) 
x2 = np.arange(X_set[:,1].min(), X_set[:,1].max(), step=0.01) 

X1, X2 = np.meshgrid(x1, x2) #generates the grid

Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)


plt.contourf(X1, X2, Z.reshape(X1.shape), alpha =0.55, cmap = ListedColormap(('red', 'green', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())


for j in np.unique(y_set):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green', 'blue'))(j), label = j)

plt.title('KNN (Training set)') 
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width') 
plt.legend()
plt.show()

#Visualizing the Testing set results
X_set, y_set = X_test, y_test

x1 = np.arange(X_set[:,0].min(), X_set[:,0].max(), step=0.01) 
x2 = np.arange(X_set[:,1].min(), X_set[:,1].max(), step=0.01) 

X1, X2 = np.meshgrid(x1, x2) #generates the grid

Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T) #z is the response


plt.contourf(X1, X2, Z.reshape(X1.shape), alpha=0.55, cmap = ListedColormap(('red', 'green', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())


for j in np.unique(y_set):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green', 'blue'))(j), label=j)
    
plt.title('KNN (Testing set)') 
plt.xlabel('Sepal Length') 
plt.ylabel('Sepal Width') 
plt.legend()
plt.show()





#SVM

#Creating classifier here
from sklearn.svm import SVC 
classifier1 = SVC(kernel = 'linear')

#Fitting classifier to the Training set
classifier1.fit(X_train, y_train)

#Predicting the Test set results
y_pred1 = classifier1.predict(X_test)

#Making the Confusion Matrix
cm1 = confusion_matrix(y_test, y_pred1)

#Visualizing the Training set results
X_set1, y_set1 = X_train, y_train

x11 = np.arange(X_set1[:,0].min(), X_set1[:,0].max(), step = 0.01) 
x21 = np.arange(X_set1[:,1].min(), X_set1[:,1].max(), step = 0.01) 

X11, X21 = np.meshgrid(x11, x21) #generates the grid

Z1 = classifier1.predict(np.array([X11.ravel(), X21.ravel()]).T) 


plt.contourf(X11, X21, Z1.reshape(X11.shape), alpha = 0.55, cmap = ListedColormap(('red', 'green', 'blue')))

plt.xlim(X11.min(), X11.max())
plt.ylim(X21.min(), X21.max())


for j in np.unique(y_set1):
    plt.scatter(X_set1[y_set1 == j, 0], X_set1[y_set1 == j, 1], c=ListedColormap(('red', 'green', 'blue'))(j), label=j)
    
plt.title('SVC (Training set)')
plt.xlabel('Sepal Length') 
plt.ylabel('Sepal Width')
plt.show()



#Visualizing the Testing set results
X_set1, y_set1 = X_test, y_test

x11 = np.arange(X_set1[:,0].min(), X_set1[:,0].max(), step = 0.01) 
x21 = np.arange(X_set1[:,1].min(), X_set1[:,1].max(), step = 0.01) 

X11, X21 = np.meshgrid(x11, x21) 

Z1 = classifier1.predict(np.array([X11.ravel(), X21.ravel()]).T) 


plt.contourf(X11, X21, Z1.reshape(X11.shape), alpha = 0.55, cmap = ListedColormap(('red', 'green', 'blue')))

plt.xlim(X11.min(), X11.max())
plt.ylim(X21.min(), X21.max())


for j in np.unique(y_set1):
    plt.scatter(X_set1[y_set1 == j, 0], X_set1[y_set1 == j, 1], c=ListedColormap(('red', 'green', 'blue'))(j), label=j)
    
plt.title('SVC (Testing set)') 
plt.xlabel('Sepal Length')  
plt.ylabel('Sepal Width') 
plt.legend()
plt.show()
#How many observations are in this data set ?
#A)150
# How many features are in this data set ?
#A)2
# Please compare the confusion matrix of both KNN and Linear SVM. Which algorithm
#get a better confusion matrix ?
#A)SVM has a better matrix as the true positives and negetives are more in confusion matrix(SVM) than in confusion matrix(KNN)
