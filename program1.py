 p# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:03:56 2018

@author: arfas
"""
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split

dataset =pd.read_csv('Data.csv')
dataset= pd.get_dummies(dataset,columns=['Country'])


x=dataset.iloc[:,[1,3,4]].values;
y=dataset.iloc[:,2];

imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer= imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)