# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 13:55:33 2018

@author: arfas
"""
#Name:Mohammed Arfa Adeeb
#ID:999993905
#Description: K-Means and Hierarchical CLustering 
#----------------------------------------K-MeansClustering------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Importing dataset
dataset=pd.read_csv('3D_network.txt')
dataset.columns = ['OSM_ID', 'LONGITUDE', 'LATITUDE', 'ALTITUDE']
X=dataset.iloc[:,[1,3]].values
#Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()

#Fitting K-Means to the dataset
kmeans=KMeans(n_clusters =3,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(X)

#Visiualising the clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=50,c='red',label='Cluster 1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=50,c='blue',label='Cluster 2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=50,c='green',label='Cluster 3')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='yellow',label='clustercenter')

plt.title('Cluster of coordinates')
plt.xlabel('Longitude')
plt.ylabel('Altitude')
plt.legend()
plt.show()

#-----------------------------------------Hierarchical CLustering----------------------------------
import matplotlib.pyplot as plt
import pandas as pd
#importing the datset
dataset=pd.read_csv('3D_network.txt')
dataset.columns = ['OSM_ID', 'LONGITUDE', 'LATITUDE', 'ALTITUDE']
X=dataset.iloc[:,[1,3]].values
#Using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendogram')
plt.xlabel('Longitude')
plt.ylabel('Eucledian Distance')
plt.show()
#Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc =hc.fit_predict(X)
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=50,c='red',label='Cluster 1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=50,c='blue',label='Cluster 2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=50,c='green',label='Cluster 3')

plt.title('Cluster of coordinates')
plt.xlabel('Longitude')
plt.ylabel('Altitude')
plt.legend()
plt.show()
#How many observations are in this data set ?
#A)999
# How many clusters you got by using K-Means ? How many clusters you got by using
#hierarchical clustering ? How you pick the number of clusters ?
#A)3 clusters for k means.3 for hierarchical clustering.The clusters picked were using elbow method and dendogram. 

