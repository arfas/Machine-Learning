# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 22:00:44 2018

@author: arfas
"""
#Name:Mohammed Arfa Adeeb
#ID:999993905
#Description:apply Artificial Neural Network we discussed during the class to the following handwritten digital numbers
import numpy as np
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train  = np.reshape(X_train, (60000, 784))
X_test  = np.reshape(X_test , (10000, 784))

X_train = np.divide(X_train, 255)
X_test = np.divide(X_test, 255)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

import keras
from keras.models import Sequential # Initialize the model
from keras.layers import Dense # Build layers of the model

classifier = Sequential()

# Adding the input layer and first hidden layer
classifier.add(Dense(output_dim = 393, init = 'uniform', activation = 'relu', input_dim = 784))

classifier.add(Dense(output_dim = 197, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 99, init = 'uniform', activation = 'relu'))


classifier.add(Dense(output_dim = 26, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss='mean_squared_error', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch=50)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.95)
