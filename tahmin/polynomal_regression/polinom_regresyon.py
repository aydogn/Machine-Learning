#KÜTÜPHANELER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#VERİ
veriler = pd.read_csv('Maaslar.csv')
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
#NUMPY DİZİ (ARRAY) DÖNÜŞÜMÜ
X=x.values
Y=y.values


#LİNEER REGRESYON
from sklearn.linear_model import LinearRegression
lr1=LinearRegression()
lr1.fit(X,Y)


#Polinomsal regresyon sınırları belli olan bir veri seti için uygun bir
# algoritmadır. Ancak veri setinin sınırlarının dışından gelen yeni veriler
# için hatası yüksek tahminlerde bulunabilir.

####POLYNOMİAL REGRESYON
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
x_poly = pr.fit_transform(X)
lr2=LinearRegression()
lr2.fit(x_poly,y)


#Derece arttıkça daha doğru değerler olabilir ama overfitting görüle debilinir.
pr3=PolynomialFeatures(degree=4)
x_poly3 = pr3.fit_transform(X)
lr3=LinearRegression()
lr3.fit(x_poly3,y)


#GÖRSELLEŞTİRME
plt.scatter(X,Y,color="red")
plt.plot(x,lr1.predict(X),color="blue")
plt.show()

plt.scatter(X,Y,color="red")
plt.plot(x,lr2.predict(x_poly),color="blue")
plt.show()

plt.scatter(X,Y,color="red")
plt.plot(x,lr3.predict(x_poly3),color="blue")
plt.show()


#TAHMİN
print("------------------------")
print(lr1.predict(np.array([[10]])))
print(lr1.predict(np.array([[6.6]])))

print("------------------------")
print(lr2.predict(pr.fit_transform(np.array([[10]]))))
print(lr2.predict(pr.fit_transform(np.array([[6.6]]))))

print("------------------------")
print(lr3.predict(pr3.fit_transform(np.array([[10]]))))
print(lr3.predict(pr3.fit_transform(np.array([[6.6]]))))

print("------------------------")