#Kütüphane
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#VERİ
#kontrollü veri seti oluşturuyoruz

#küme1
x1 = np.random.normal(25,5,20) #ortalama:25, standart sapma:5
y1 = np.random.normal(25,5,20)

#küme2
x2 = np.random.normal(55,5,20) 
y2 = np.random.normal(60,5,20)

#küme1
x3 = np.random.normal(55,5,20) 
y3 = np.random.normal(15,5,20)

#birleştir
x = np.concatenate((x1,x2,x3), axis=0)
y = np.concatenate((y1,y2,y3), axis=0) 

dictionary = {"x":x, "y":y}

data = pd.DataFrame(dictionary)


#Görselleştirme
plt.figure()
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.scatter(x3, y3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Hiyerarşik Kümeleme Yöntemi için Oluşturulan Veri Seti")
plt.show()
#algoritma veriyi bizim gördüğümüz gibi renklere ayrılmış bir şekilde değil
# aksine aynı renkte gibi gibi görecek 


# DENDOGRAM
from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(data, method="ward")
dendrogram(merg, leaf_rotation=(90))
plt.xlabel("Veri Noktaları")
plt.ylabel("Öklid Mesafesi") 
plt.show()
#üzümden salkım keser gibi en uzun çizgiden en uygun öklid mesafesini kesin 


#Eğitim ve Test
from sklearn.cluster import AgglomerativeClustering
hiyerarsi_kume = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
kume = hiyerarsi_kume.fit_predict(data)
#fit.predict her bir noktaya ilişkin hangi öbeğe dahil olduğuna dair indeks verir

data["label"] = kume

plt.figure()
plt.scatter(data.x[data.label == 0], data.y[data.label == 0], color="red", label="Kume 1")
plt.scatter(data.x[data.label == 1], data.y[data.label == 1], color="green", label="Kume 2")
plt.scatter(data.x[data.label == 2], data.y[data.label == 2], color="blue", label="Kume 3")

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("3-Ortalama Küme Sonucu")
plt.show()








