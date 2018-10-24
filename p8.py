# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:07:24 2018

@author: arfas
"""

from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

lfw_dataset = fetch_lfw_people(min_faces_per_person = 100)
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_lfw_people



# Load datalfw_dataset = fetch_lfw_people(min_faces_per_person=100)
_, h, w = lfw_dataset.images.shape

X = lfw_dataset.data

y = lfw_dataset.target

target_names = lfw_dataset.target_names

# split into a training and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from keras.utils import np_utils
y_train_new = np_utils.to_categorical(y_train)

y_test_new = np_utils.to_categorical(y_test)

# Compute a PCA 

n_components = 100

pca = PCA(n_components=n_components, whiten=True).fit(X_train)

# apply PCA transformation to training data

X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)



import keras
from keras.models import Sequential ##initialize your model
from keras.layers import Dense ## build layers of the model

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu', input_dim = 100))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 25, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'softmax'))

# Compiling the ANN
#classifier.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['accuracy'])
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train_pca, y_train_new , batch_size = 10, nb_epoch = 50)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test_pca)
#y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test_new, y_pred)
#print(cm)