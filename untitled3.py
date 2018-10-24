# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 23:57:09 2018

@author: arfas
"""

import numpy as np

py1=0.33
py2=0.67 

m1=[[5.01],[3.42]] 
m2=[[6.26],[2.87]] 

m1=np.array(m1) 
m2=np.array(m2) 

cm1=[[0.122, 0.098],[0.098, 0.142]] 
cm1=np.array(cm1) 
cm2=[[0.435, 0.121],[0.121, 0.110]] 
cm2=np.array(cm2) 

modcm1=np.linalg.det(cm1) 
modcm2=np.linalg.det(cm2) 

xt=[[6.75],[4.25]] 
xt=np.array(xt) 

mtsub1=np.subtract(xt, m1) 
mtsub2=np.subtract(xt, m2) 


f1=((1/((np.sqrt(2*np.pi))*np.power(modcm1, (1/2))))(np.exp(-((np.matmul(np.matmul(mtsub1.transpose(), (1/cm1)), mtsub1))/2))))
f2=((1/((np.sqrt(2*np.pi))*np.power(modcm2, (1/2))))(np.exp(-((np.matmul(np.matmul(mtsub2.transpose(), (1/cm2)), mtsub2))/2))))

#calculates posterior probability for class c1
Ppop1= (f1*py1)/((f1*py1)+(f2*py2))

#calculates posterior probability for class c2
Ppop2= (f2*py2)/((f1*py1)+(f2*py2))