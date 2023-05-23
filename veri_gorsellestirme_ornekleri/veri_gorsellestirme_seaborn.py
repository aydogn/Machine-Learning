# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:12:47 2023

@author: aydogn
"""

import matplotlib.pyplot as plt
import pandas as pd

#seaborn kategorikte çok kolaylık bir kütüphanedir
import seaborn as sns

import warnings 
warnings.filterwarnings("ignore")

veri=pd.read_csv("olimpiyatlar_temizlenmis.csv")


#Saçılım Grafiği
"""
def sacilimGrafikCiz():
    sns.set_style('darkgrid') #Grid arka planını belirler
    sns.scatterplot(x='boy', y='kilo', data=veri)
    plt.title('Boy ve Ağırlık ilişkisi - Beyaz Izgara Tema')
    plt.show()
    
sacilimGrafikCiz()


#hue neye göre renklendirme yapacağı belirtilir
#style neye göre şekilleneceğini belirtir
#size neye göre boyutlanacığını belirtir sizes(en küçük boyut, en bü bo)
#palette: önceden düzenlenmiş belirli renk paletlerini otomatik atar

#plotlib'te bu kategorik değişkenleri tek tek ayırıp işlem yapılması gerekirdi
#ama seabornda tek parametreyle bu iş halledildi
sns.scatterplot(x='boy', y='kilo', data=veri, sizes=(15,200), palette='Set1',
                hue='madalya', style='cinsiyet', size='yas')
plt.title('Madalyaya Göre Boy ve Ağırlık Dağılımı')
plt.show()


#regresyon plotu saçılım grafiğini oluşturalım
sns.regplot(x='boy', y='kilo', data=veri, marker='+',
            scatter_kws={'alpha': 0.2}) #alpha: transparan parametresi
plt.title('Boy ve Kilo Dağılımı')
plt.show()
"""


#Çizgi Grafiği
"""
sns.lineplot(x='boy', y='kilo', data=veri, hue='cinsiyet')
plt.title('Cinsiyete Göre Boy ve Kilo Dağılımı')
plt.show()    
"""


#Histogram
"""
sns.displot(veri, x='kilo', hue="cinsiyet")
plt.ylabel('Frekans')
plt.title('Ağırlık Histogramı')
plt.show()

#iki tane yan yana
sns.displot(veri, x='kilo', col="cinsiyet", multiple='dodge')
plt.show()

#2 boyutlu 

#kind='kde': kernel density
#buna hue='cinsiyet' eklersek kadın-erkek için ayrı çizdiririz
# sns.displot(veri, x='kilo', y='boy', kind="kde")
# plt.xlabel('Ağırlık')
# plt.ylabel('Boy')
# plt.title('Ağırlık Boy Histogramı')
# plt.show()

#kdeplotla önceki histogramı yazabiliriz 
#burda True diyerek farkli bir görünüm kazandırdık ve hue='cinsiyet' ekledik
sns.kdeplot(data=veri, x='kilo', y='boy', hue="cinsiyet", fill=True)
plt.xlabel('Ağırlık')
plt.ylabel('Boy')
plt.title('Cinsiyete Göre Ağırlık Boy Histogramı')
plt.show()
"""


#Çubuk Grafiği
"""
sns.barplot(x="madalya", y="boy", hue=('cinsiyet'), data=veri)
plt.title("Cinsiyete Göre Madalyaya-Boy Grafikleri")
plt.show()


#col='sezon' sezona göre ayrı ayrı yan yana plot çizer
#kind ile tür belirlenir
#height ve aspect grafiğin fiziksel özelliğiyle ilgilidir; en, boy
sns.catplot(x="madalya", y="boy", hue='cinsiyet', col='sezon',
            kind='bar', data=veri, height=4, aspect=0.7)
plt.show()
"""


#Kutu Grafiği
"""
sns.boxplot(x="sezon", y="boy", hue="cinsiyet", data=veri, palette="Set2")
plt.show()


veri_gecici = veri.loc[:,["yas","kilo","boy"]]
sns.boxplot(data=veri_gecici, orient="h", palette="Set2")
plt.show()


#Farklı madalyalar için yaz ve kış sezonlarında erkek ve kadınlar arasında boy
#farklılıkları
sns.catplot(x="sezon", y="boy", hue="cinsiyet",col="madalya", data=veri,
            kind="box", height=4, aspect=0.7, palette="Set2")
plt.show()
"""


#Isı Haritası
"""
#veri.corr()=verideki değişkenler arasındaki ilişkiyi gösterir
#annot=True içine verilerin yazısını ekler
#fmt=yazılacak oranların ',' den sonra kaç basamak olacağını belirler
sns.heatmap(veri.corr(), annot=True, linewidths=0.5, fmt='.3f')
plt.show()
"""


#Keman Grafiği
"""
#split=False ayrı ayrı simetrik görsel çıkarır
sns.violinplot(x="sezon", y="boy",hue="cinsiyet", data=veri, split=True)
plt.show()


sns.catplot(x="sezon", y="boy", hue="cinsiyet", col="madalya",data=veri,
            kind="violin", split=True, height=4, aspect=0.7)
plt.show()
"""


#Ortaklık Grafiği
"""
# sns.jointplot(data=veri, x="kilo", y="boy", hue="sezon", kind='kde')
# plt.show()


g= sns.JointGrid(data=veri, x="kilo", y="boy")
g.plot_joint(sns.histplot)
g.plot_marginals(sns.violinplot)  
plt.show()
"""


#Çift/Eş Grafiği
#stünlar arasında ikili bir şekilde grafik oluşturur
"""
sns.pairplot(veri)
plt.show()
#burada istemediğimiz grafikler var istediğimiz grafikleri yerleştirmek için..

#burada istediklerimizi yerleştirdik
g=sns.PairGrid(veri)
g.map_upper(sns.histplot) #upper matrix tarafına yerleştirilenler
g.map_lower(sns.kdeplot, fill=True)#lower matrix tarafına yerleştirilenler
g.map_diag(sns.histplot, kde=True)#diagonal tarafına yerleştirilenler
"""


#Sayma Grafiği
"""
sns.countplot(x='sehir', data=veri)
plt.xticks(rotation=90)
plt.show()
"""

















