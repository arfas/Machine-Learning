# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 16:31:07 2018

@author: arfas
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the datset
dataset=pd.read_csv('3D_spatial_network.csv')
X=dataset.iloc[:,[1,3]].values
#Using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Eucledian Distance')
plt.show()
#Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc =hc.fit_predict(X)
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=50,c='red',label='Cluster 1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=50,c='blue',label='Cluster 2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=50,c='green',label='Cluster 3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=50,c='cyan',label='Cluster 4')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=50,c='magenta',label='Cluster 5')
plt.title('CLuster of Cuatomers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()

