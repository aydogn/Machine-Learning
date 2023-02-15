#KÜTÜPHANELER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#VERİ
veriler = pd.read_csv('Maaslar.csv')
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

X=x.values
Y=y.values


#Bağımsız değişkenleri bilgi kazancına göre aralıklara ayırıyor. Tahmin
# esnasında bu aralıktan bir değer sorulduğunda cevap olarak bu aralıktaki
# (eğitim esnasında öğrendiği) ortalamayı söylüyor. Dezavantajı overfittingtir.

#DECISION TREE
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)


#GÖRSELLEŞTİRME
plt.scatter(X,Y,color="red")
plt.plot(X,r_dt.predict(X),color="blue")
plt.show()


#TAHMİNİ SAYISAL DEĞERLER
print("------------------------")
print("DECISION TREE")
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))
print("------------------------")