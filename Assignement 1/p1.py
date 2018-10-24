# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:19:31 2018

@author: arfas
"""
#Name:Mohammed Arfa Adeeb
#ID:999993905
#Description:The purpose of the program is to read the dataset from a file,then replace the ? to Nan 
#Then clean the data using imputer then Split the dataset into training features, test features, training response, test response

import pandas as pd #imports the package panda
from sklearn.preprocessing import Imputer #importes importer class
from sklearn.cross_validation import train_test_split #imports train_test_split class
import numpy
dataset =pd.read_csv('arrhythmia.data.csv',header=None) #use the method read_csv to read data set
dataset = dataset.replace('?',numpy.NaN)#replaces all ? to NaN by using replace
print(dataset)
x=dataset.iloc[:,:-1];#this is the input to the file
y=dataset.iloc[:,-1];#this gives the response
imputer = Imputer(missing_values= 'NaN', strategy='mean', axis=0)#use imputer class to replace all NaN missing values for the data with column
imputer.fit(x)
data_clean = imputer.transform(x)
mydata = pd.DataFrame(data_clean)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)#Splits your dataset into training features, test features, training response, test response
#DATA MATRIX
#there are 452 observations in the dataset
#there are 280 features in the dataset
#the value of (Y) is the response