#buna bi daha bak!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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


#Support Vector Regression (SVR) gibi bazı makine öğrenme algoritmaları,
# verilerin ölçeklerinden etkilenebilir ve ölçekler arasındaki farklılıklar
# nedeniyle performansı bozabilir. Standart Scaler, verileri ortalama değerine
# ve standart sapmasına göre ölçeklendirerek bu problemi çözmek için
# kullanılabilir.

#SUPPORT VECTOR REGRESSION
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
sc2=StandardScaler()

x_olcekli = sc1.fit_transform(X)
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))


#Destek vektör regresyonu uyguladığımızda, çizeceğimiz aralığın maksimum
# noktayı içerisine almasını sağlamaktır.

from sklearn.svm import SVR
svr_reg = SVR(kernel="rbf",degree=7,C=10)
svr_reg.fit(x_olcekli,y_olcekli)



#GÖRSELLEŞTİRME
plt.scatter(x_olcekli,y_olcekli,color="red")
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color="blue")
plt.show()

#TAHMİN
#Burada tahmin yaparken dikkat edilmesi gereken husus tahmin değerini scale
# etmek gerektiğidir. işin sonunda da scale edilen değeri tekrardan normal hale
# döndürmek anlamamızı kolaylaştırır. Not: Burda bunu yapmadık. 
print(svr_reg.predict(np.array([[1.2]])))
print(svr_reg.predict(np.array([[6.6]])))
print(sc2.inverse_transform(svr_reg.predict(sc1.transform(np.array([[11]]).reshape(1,-1)))))
#y_pred=sc2.inverse_transform(svr_reg.predict(sc1.transform(np.array([[11]]))))

