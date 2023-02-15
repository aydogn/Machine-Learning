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


#karar ağaçlarının en büyük problemlerinden biri aşırı öğrenme-veriyi
# ezberlemedir (overfitting). Rassal orman modeli bu problemi çözmek için
# 100'lerce karar ağacı oluşturuluyor ve her bir karar ağacı bireysel olarak
# tahminde bulunuyor ve sonuçların ortalaması alınıyor.

#3.5 RANDOM FOREST REGRESSION
from sklearn.ensemble import RandomForestRegressor
# n_estimators -> kaç tane decision tree kullanacağını verir.
# En doğru sonuç için arttırmak çözüm değildir arttırınca değerler 
#sabitleşir.
rf_reg = RandomForestRegressor(n_estimators=100,random_state=0)
rf_reg.fit(X,Y.ravel())


###4 GÖRSELLEŞTİRME
plt.scatter(X,Y,color="red")
plt.plot(X,rf_reg.predict(X),color="blue")
plt.show()


###5 TAHMİNİ SAYISAL DEĞERLER
print("------------------------")
print("RANDOM FOREST REGRESSION")
print(rf_reg.predict([[10]]))
print(rf_reg.predict([[6.6]]))
print("------------------------")













