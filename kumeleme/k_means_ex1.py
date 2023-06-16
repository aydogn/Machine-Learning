import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:,3:].values


#KMeans
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3,init='k-means++')
kmeans.fit(X)

sonuclar=[]

print(kmeans.cluster_centers_)

for i in range(1,11):
    
#random_state kısmına bi değer yazıyorumki değikenim sadece n_cluster olsun
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=(12))
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    
plt.plot(range(1,11),sonuclar)

#Bu döngü sayesinde en uygun n_cluster buluyoruz
#Dirsek noktası dediğimiz noktayı tercih ediyoruz burada 2,3 ve 4 uygundur

kmeans2 = KMeans(n_clusters= 4, init='k-means++', random_state=(12))
Y_predictKM=kmeans2.fit_predict(X)
print(Y_predictKM)
plt.scatter(X[Y_predictKM==0,0], X[Y_predictKM==0,1], s=100, c='red')
plt.scatter(X[Y_predictKM==1,0], X[Y_predictKM==1,1], s=100, c='blue')
plt.scatter(X[Y_predictKM==2,0], X[Y_predictKM==2,1], s=100, c='green')
plt.scatter(X[Y_predictKM==3,0], X[Y_predictKM==3,1], s=100, c='yellow')
plt.title('KMeans')
plt.show()


#Agglomerative Clustirng
from sklearn.cluster import AgglomerativeClustering
ac= AgglomerativeClustering(n_clusters=4, linkage='ward', affinity='euclidean')
Y_predictAC=ac.fit_predict(X)
print(Y_predictAC)

plt.scatter(X[Y_predictAC==0,0], X[Y_predictAC==0,1], s=100, c='red')
plt.scatter(X[Y_predictAC==1,0], X[Y_predictAC==1,1], s=100, c='blue')
plt.scatter(X[Y_predictAC==2,0], X[Y_predictAC==2,1], s=100, c='green')
plt.scatter(X[Y_predictAC==3,0], X[Y_predictAC==3,1], s=100, c='yellow')
plt.title('AC')
plt.show()


import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()





