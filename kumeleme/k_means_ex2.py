#Kütüphane
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#VERİ
#kontrollü veri seti oluşturuyoruz

#küme1
x1 = np.random.normal(25,5,1000) #ortalama:25, standart sapma:5
y1 = np.random.normal(25,5,1000)

#küme2
x2 = np.random.normal(55,5,1000) 
y2 = np.random.normal(60,5,1000)

#küme1
x3 = np.random.normal(55,5,1000) 
y3 = np.random.normal(15,5,1000)

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
plt.title("K-ortalama Yöntemi için Oluşturulan Veri Seti")
plt.show()
#algoritma veriyi bizim gördüğümüz gibi renklere ayrılmış bir şekilde değil
# aksine aynı renkte gibi gibi görecek 


#K-means (en uygun öbek sayısını bulun)

from sklearn.cluster import KMeans
wcss=[]

#sırayla k değerlerini dene
for k in range(1,15):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(data) #gözetimsiz öğrenme olduğu için train test kullanmıyoruz
    wcss.append(kmeans.inertia_)#k değerlerinin ortalamasını listeye ekle

plt.figure()
plt.plot(range(1,15), wcss)
plt.xticks(range(1,15))
plt.xlabel("Küme Sayısı (K)")
plt.ylabel("wcss")
plt.show()
#Burada dirsek noktamız var bu nokta en dar açı neyse oraya bakaraktan bulunur


#Eğitim ve Test
k_ortalama = KMeans(n_clusters=3)
kumeler = k_ortalama.fit_predict(data)
#fit.predict her bir noktaya ilişkin hangi öbeğe dahil olduğuna dair indeks verir

data["label"] = kumeler

plt.figure()
plt.scatter(data.x[data.label == 0], data.y[data.label == 0], color="red", label="Kume 1")
plt.scatter(data.x[data.label == 1], data.y[data.label == 1], color="green", label="Kume 2")
plt.scatter(data.x[data.label == 2], data.y[data.label == 2], color="blue", label="Kume 3")
plt.scatter(k_ortalama.cluster_centers_[:,0], k_ortalama.cluster_centers_[:,1], color="yellow")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("3-Ortalama Küme Sonucu")
plt.show()






                    