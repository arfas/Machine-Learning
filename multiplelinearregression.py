# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:15:31 2018

@author: arfas
"""
#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('50_Startups.csv')

X=dataset.iloc[:,:-1]
y=dataset.iloc[:,4].values
#Creating  dummy variable
X=pd.get_dummies(X,columns=['State']).values
#Avoiding the dummy variable trap
X=X[:,:-1]
#SPlitting the dataset into the trianing and testing models
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#Fitting Multiple Linear Regression to the Traning set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
#Predecting the Test set results
y_pred=regressor.predict(X_test)
#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]


regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,1,2,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,1,2,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,3]] #Adj. R-squared=0.948
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1]] #Adj. R-squared=0.945
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#Automatic implementation of Backward Elimination
import statsmodels.formula.api as sm
def backwardElimination(x,sl):
    numVars=len(x[0])
    for i in range(0,numVars):
        regressor_OLS=sm.OLS(y,x).fit()
        maxVar=max(regressor_OLS.pvalues).astype(float)
        if maxVar >sl:
            for j in range(0,numVars -i):
                if(regressor_OLS.pvalues[j].astype(float)==maxVar):
                   x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

SL=0.05
X_opt = X[:,[0,1,2,3,4,5]]
X_Modeled=backwardElimination(X_opt,SL)






