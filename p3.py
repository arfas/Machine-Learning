# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:11:47 2018

@author: arfas
"""
#Name:Mohammed Arfa Adeeb
#ID:999993905
#Description:The purpose of the program is to import,split and to predict the test data.
#Also to build the optimal model using Backward Elimination.
#import the libraries
import numpy as np
import pandas as pd


#importing the dataset
dataset=pd.read_csv('Advertising.csv')

X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1].values

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
X=np.append(arr=np.ones((200,1)),values=X,axis=1)


X_opt=X[:,[0,1,2,3]]#Adj. R-squared=0.897
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,2,3]] #Adj. R-squared=0.897
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
# How many observations are in this data set ?
#A)There are 200 observations in dataset
# How many features are in this data set ?
#A)There are 5 feautres in dataset
# What is the response for this data set ?
#A)Y is the response for the dataset
# Which predictors are the most significant for this dataset ? Please explain Why ?
#A)The coefficients with less than 0.05 value in this case [0,2,3] predictors.
#As the p value increases Rsquared changes its value which is not recommended.


