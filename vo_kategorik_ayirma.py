###KATEGORİK VERİLERİ NUMERİK YAPMA###

#kütüphane
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri
veriler = pd.read_csv('eksikveriler.csv')
ulke = veriler.iloc[:,0:1].values


from sklearn import preprocessing
#Bu işlemle LabelEncoder() fit_transformla birlikte önce verileri alfabetik
#şekilde sıralıyoruz sonra her birine sırasına göre 0 1 2 .. veriyoruz
#fr:0 tr:1 us:2 aldı.

le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)

#OneHotEncoder()'la bizler önceki aldığımız ve sayılara çevirdiğimiz
#verileri 1 ve 0 lara çeviren bir fonksiyon bunu yapmak için de kaç tane
#veri varsa o kadar sutün oluşturuyor ve o sutünlara varsa 1 yoksa 0 
#yazıyor.

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)



