# -*- coding: utf-8 -*-
"""
Created on Wed May  9 01:55:43 2018

@author: arfas
"""

a=1000000000
for i in range(100000):
    a = a + 1e-6
print (a - 1000000000)