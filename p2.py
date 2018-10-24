# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:16:17 2018

@author: arfas
"""
#Name:Mohammed Arfa Adeeb
#ID:999993905
#Description:The purpose of the program is to import the dataset,split the dataset,Fitting single linear regression
#predecting the test set results,visiualsing the training set results and visiualising the test set results.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset=pd.read_excel('autoInsurance.xls')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
#splitting the datset into training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
#Predicting the Test set results
y_pred=regressor.predict(X_test)
#Visualising the Training set results
plt.scatter(X_train,y_train,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title('No of claims vs Total Payment for all claims (Training set)')
plt.xlabel('No of claims')
plt.ylabel('Total Payment for all claims')
plt.show()
#Visiualising the Test set Results
plt.scatter(X_test,y_test,color='green')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title('No of claims vs Total Payment for all claims (Test set)')
plt.xlabel('No of claims')
plt.ylabel('Total Payment for all claims')
plt.show()
#Describe the data matrix
#How many observations are in this data set ?
#A)62
#How many features are in this data set ?
#A)1
#What is the response for this data set ?
#A)Y is the response for the dataset.