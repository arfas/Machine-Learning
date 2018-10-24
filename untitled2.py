# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:20:32 2018

@author: arfas
"""


import matplotlib.pyplot as plt
from keras.datasets import mnist
(X_train,y_train),(X_test,y_test)=mnist.load_data()
plt.subplot(221)
plt.imshow(X_train[1],cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[3],cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[5],cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[7],cmap=plt.get_cmap('gray'))

plt.show()
import matplotlib.pyplot as plt
from keras.datasets import mnist
(X_train,y_train),(X_test,y_test)=mnist.load_data()
plt.subplot(221)
plt.imshow(X_train[2],cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[0],cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[13],cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[15],cmap=plt.get_cmap('gray'))

plt.show()
import matplotlib.pyplot as plt
from keras.datasets import mnist
(X_train,y_train),(X_test,y_test)=mnist.load_data()
plt.subplot(221)
plt.imshow(X_train[17],cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[19],cmap=plt.get_cmap('gray'))


plt.show()
