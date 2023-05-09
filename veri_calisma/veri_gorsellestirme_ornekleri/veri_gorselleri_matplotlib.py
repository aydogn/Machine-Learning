# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 15:17:39 2023

@author: aydogn
"""

import matplotlib.pyplot as plt
import numpy as np

#Plot fonksiyonu kullanımı Çizgi Grafiği
"""
plt.plot([1,2,3,4], [1,4,9,16], color='red', marker='o', linestyle='dashed',
         linewidth=2, markersize=12) 
plt.xlabel('tamsayılar')
plt.ylabel('tamsayıların kareleri')
plt.title('Bazı Tam Syıların Kareleri')
plt.show()

#başka kısa yolu

t = np.arange(0.0, 5.0, 0.1)

#plot fonksiyonun ilk 3 değeri x,y,(renk şekil) sonrakiler 3 katı olarak gider
plt.plot(t, t, 'r>--', t, t**2, 'bs', t, t**3, 'g^') #'bs' blue square
plt.show()

#e sayısı ve cos kullanım örneği
plt.plot(t, (np.exp(-t) * np.cos(2*np.pi*t)), 'k',
         t, (np.exp(-t) * np.cos(2*np.pi*t)), 'bo')
plt.axis([0,10,-1,1])#grafiği rahat görmek için x ekseninde 0'a 10 y'de -1'e 1
plt.show()
"""


#Saçılım Grafiği 
"""
veri={'a': np.arange(50),
      'c': np.random.randint(0,50,50),
      'd': np.random.randn(50)}

veri['b'] = veri['a'] + (10 * np.random.randn(50))
veri['d'] = np.abs(veri['d']) * 100

#c: noktaları verilere göre renklendirir default değeri mavidir
#s: noktaları verilere göre boyutlandırır
plt.scatter('a', 'b', c='c', s='d', data= veri)
plt.xlabel('a stünundaki veriler')
plt.ylabel('b stünundaki veriler')
plt.show()
"""


#Yan yana birden fazla grafik çizme
"""
isimler=['Ahmet', 'Mehmet', 'Ayşe', 'Fatma']
degerler=[5, 25, 50, 100]

plt.figure(figsize=(12,3)) #ana figürün boyutu 

plt.subplot(131) #1 satır ve 3 stünluk yerin 1. lokasyonuna yerleştir
plt.bar(isimler, degerler)

plt.subplot(132) #1 satır ve 3 stünluk yerin 2. lokasyonuna yerleştir
plt.scatter(isimler, degerler)

plt.subplot(133) #1 satır ve 3 stünluk yerin 3. lokasyonuna yerleştir
plt.plot(isimler, degerler)

plt.subtitle('Birçok Farklı Grafik')
plt.show()
"""


#Histogram
"""
#ortalaması 100, standart sapması 15 olan 10000 adet veri
mu1 = 100
sigma1 = 15
x = mu1 + (sigma1 * np.random.randn(10000))

#ortalaması 110, standart sapması 10 olan 10000 adet veri
mu2 = 110
sigma2 = 10
y = mu2 + (sigma2 * np.random.randn(10000))

#Söz konusu verilerin histogramı

#100 adet bins mavi/yeşil renkli 0.25 şeffaf
n, bins, patches = plt.hist(x, 100, density=1, facecolor='b', alpha=0.25)
n, bins, patches = plt.hist(y, 100, density=1, facecolor='g', alpha=0.25)

plt.xlabel('Veriler')
plt.ylabel('Olasılıklar')
plt.title(r'Verilerin Histogramı: $\mu_1=100,\ \mu_2=110$')#r ve $ ilev içine 
#özel yazı yazacağımı belirttim mu_1'deki _ ifadesi 1'i alta yaz anlamında
#başına \ mu_1'i direk almaması için $ latektedekine benzer kullanımdadır 

plt.annotate('Gauss eğrisinin tepesi', xy=(110,0.04), xytext=(130,0.045),
             arrowprops=dict(facecolor='black', shrink=0.1))
# Burada bir ok oluşturuyorum ilkine ne yazması gerektiği xy ile nereyi
#  göstereceğini xytext ile ok yazısı nerede olacağı arrowprops'la okun
#  şeklini, rengini ayarlıyorum.

#verilerin ne verisi olduğunu anlatmada kolaylık olması için text koyuyoruz
plt.text(75, 0.025, r'$\mu_1=100,\ \sigma_1=15$')
plt.text(120, 0.035, r'$\mu_1=100,\ \sigma_1=15$')

plt.axis([40, 160, 0, 0.05])#eksen sınırlama 
plt.grid(True)
plt.show()
"""


#Lineer Logaritmik 
"""
x=np.arange(0,10,0.01)
y=np.exp(x)

plt.figure()

plt.subplot(121)
plt.plot(x,y)
plt.yscale('linear')
plt.title('Lineer')
plt.grid(True)

#Logaritmik yüksek sayılarda artışı logaritmik olanları görmeyi kolaylaştırır
plt.subplot(122)
plt.plot(x,y)
plt.yscale('log')
plt.title('Logaritmik')
plt.grid(True)

plt.show()
"""







































