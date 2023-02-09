#KÜTÜPHANE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#VERİ
veriler = pd.read_csv('satislar.csv')
print(veriler)

aylar = veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)


#EĞİTİM VE TEST İÇİN BÖLÜNME
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)



#Doğrusal regresyon, öngörülen ve gerçek çıkış değerleri arasındaki
# uyumsuzlukları en aza indiren düz bir çizgi ya da yüzeye yerleşir. Bir çift
# eşleştirilmiş veri kümesi için en uygun satırı keşfetmek üzere 'en küçük
# kareler' yöntemini kullanan basit doğrusal regresyon hesaplayıcılar vardır.
# Daha sonra, Y'den (bağımsız değişken) X'in (bağımlı değişken) değerini
# tahmin edersiniz.

###LİNEER REGRESYON###
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
tahmin = lr.predict(x_test)


#GÖRSELLEŞTİRME
x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train,y_train)
plt.plot(x_test,tahmin)
#tabloda başlıklar at
plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")












