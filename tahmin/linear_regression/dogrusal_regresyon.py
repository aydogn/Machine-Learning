#Veri seti: ay sayısına göre satışlar verisi

#KÜTÜPHANE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#VERİ
veriler = pd.read_csv('satislar.csv', sep=",")

aylar = veriler[['Aylar']]

satislar = veriler[['Satislar']]


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

tahmin_Y=lr.intercept_ #Y eksenini kestiği nokta
slope=lr.coef_ #Eğim(slope)

sonuc = lr.predict(np.array([52]).reshape(-1,1)) #52. ayın tahminini yaptık
print(sonuc[0][0])


#Eğitim ve test için bölmeye gerek kalmadan tahmin yapılabilinir.

#öncelikle predict fonksiyonu için array hazırlamamız lazım sonuna
# reshape(-1,1) yazacağız çünkü regresyon böyle istiyor.
array = np.array([65,70,75,80,85,90,95,100]).reshape(-1,1)

y_head=lr.predict(array)


#GÖRSELLEŞTİRME
x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train,y_train)
plt.plot(x_test,tahmin)
#tabloda başlıklar at
plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")

#scatter plot
plt.scatter(x_train,y_train)
plt.scatter(x_test,tahmin)
plt.show()


#TAHMİN ETTİĞİMİZ VERİLERİN GÖRSELİ
plt.plot(array,y_head)
#tabloda başlıklar at
plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
plt.grid(True)
#scatter plot
plt.scatter(array,y_head)
plt.show()

# confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = lr.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
#sıcaklık haritası
f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="white", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()




