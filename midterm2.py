# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:00:16 2018

@author: arfas
"""

import pandas as pd #imports the package panda
import numpy as np

from sklearn.preprocessing import Imputer #importes importer class

dataset =pd.read_csv('HousePrice_UK.csv',header=None)
dataset = dataset.replace('?',np.NaN)

print(dataset)
X =dataset.iloc[:,[4,3,9,22,23]];
Y= dataset.iloc[:,44];
#X=pd.get_dummies(X,columns=['Index','AveragePrice','SalesVolume','FlatPrice','FlatIndex'])

X=X[:,[4,3,9,22,23]]
imp = Imputer(missing_values= 'nan', strategy='mean', axis=0)
imp.fit(dataset)
data_clean = imp.transform(dataset)
print(data_clean)

from sklearn.cross_validation import train_test_split

X_train,X_test,Y_train,y_test =train_test_split(data_clean,Y,test_size=0.25,random_state=0)
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((118819,1)),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]


regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,1,2,3,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,1,2,3]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,3]] #Adj. R-squared=0.948
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1]] #Adj. R-squared=0.945
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()
#How many observations are in the dataser?
#A)118819
#How many features are in the dataset?
#A)1
#Which predictors are the most significant for this dataset?Please explain why?
#A)The coefficients with less than 0.05 value in this case [0,2,3] predictors.
#As the p value increases Rsquared changes its value which is not recommended.

