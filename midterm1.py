# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:15:25 2018

@author: arfas
"""

import pandas as pd
import numpy
from sklearn.preprocessing import Imputer 

from sklearn.metrics import accuracy_score




dataset =pd.read_csv('heartDisease.data.csv',header=None) 
dataset = dataset.replace('?',numpy.NaN)
print(dataset)
X =dataset.iloc[:,:-1];
Y= dataset.iloc[:,-1]
imp = Imputer(missing_values= 'NaN', strategy='mean', axis=0)
imp.fit(dataset)
data_clean = imp.transform(dataset)
print(data_clean)

from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test =train_test_split(data_clean,Y,test_size=0.25,random_state=0)



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test =sc.transform(X_test)



# fitting the classifier to the training set 
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=8,metric='minkowski',p=2)

classifier.fit(X_train,y_train)

# predicting the test set results
y_pred =classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)



#-------------------------SVC-------------------------------------------#
import pandas as pd
from sklearn.svm import SVC 




dataset = dataset.replace('?',numpy.NaN)
print(dataset)
X =dataset.iloc[:,:-1];
Y= dataset.iloc[:,-1]
imp = Imputer(missing_values= 'NaN', strategy='mean', axis=0)
imp.fit(dataset)
data_clean = imp.transform(dataset)
print(data_clean)

from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test =train_test_split(data_clean,Y,test_size=0.25,random_state=0)


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
cm1=confusion_matrix(y_test,y_pred)
print(cm1)
#How many observations are in the dataset
#A)294
#How many features are in the dataset
#A)1
#Which Algorithm has better confusion matrix
#A)SVM has a better matrix as the true positives and negetives are more in confusion matrix(SVM) than in confusion matrix(KNN)


