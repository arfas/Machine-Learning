# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:54:12 2018

@author: arfas
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:08:14 2018

@author: arfas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC




dataset=pd.read_csv('Social_Network_Ads.csv')
 
X =dataset.iloc[:,[2,3]];
Y= dataset.iloc[:,4]


from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test =train_test_split(X,Y,test_size=0.25,random_state=0)




from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test =sc.transform(X_test)



# fitting the classifier to the trfrom sklearn.svm import SVM
classifier=SVC(kernel='rbf')


classifier.fit(X_train,y_train)

# predicting the test set results
y_pred =classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
accuracies.mean()
accuracies.std()