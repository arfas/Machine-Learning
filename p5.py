# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:17:37 2018

@author: arfas
"""
#Name:Mohammed Arfa Adeeb
#ID:999993905
#Description:Calculating the postirior probabilities for c1 and c2
#import numpy and declare the matrices
import numpy as np
mu1 = np.matrix('5.01;3.42')
mu2 = np.matrix('6.26; 2.87')
cov1 = np.matrix('0.122 0.098; 0.098 0.142')
cov2 = np.matrix('0.435 0.121; 0.121 0.110')
X = np.matrix('6.75;4.25')
p1 = 0.33
p2 = 0.67
d=2

# Calculations of last part
last1 = (X-mu1)
last1Trans = np.transpose(last1)

from numpy.linalg import inv
cov1Inv = inv(cov1)

last2 = (last1Trans*cov1Inv*last1)
last = (-(last2/2))
 
# Calulations of first part
from numpy.linalg import det
first1 = ((np.sqrt(2*np.pi))**d)*(np.sqrt(det(cov1)))
first = 1/first1
 
final = first*np.exp(last)

###### calculation for py2
# Calculations of last part
last1_2 = (X-mu2)
last1_2Trans = np.transpose(last1_2)
#import inverse class
from numpy.linalg import inv
cov2Inv = inv(cov2)

last2_2 = (last1_2Trans*cov2Inv*last1_2)
last_2 = (-(last2_2/2))
 
# Calulations of first part
from numpy.linalg import det
first1_2 = ((np.sqrt(2*np.pi))**d)*(np.sqrt(det(cov2)))
first_2 = 1/first1_2
#calculation for the final part 
final_2 = first_2*np.exp(last_2)
#Probability calculation for c1 
py1_1 = (final*p1)
py1_2 = (final*p1+final_2*p2)
#Posterior probablity P(Y=1|X=x)
py1 = py1_1/py1_2

print('probability for c1: ',py1)
 
#Probabilty calculation for c2
py2_1 = (final_2*p2)
py2_2 = (final*p1+final_2*p2)
#Posterior probablity P(Y=2|X=x)
py2 = py2_1/py2_2
print('probability for c2: ',py2)

 
 