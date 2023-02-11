#1 KÜTÜPHANELER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


###2 VERİ YÜKLEME
veriler = pd.read_csv('Maaslar.csv')

#DATAFRAME DİLİMLEME(SLICE)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#NUMPY DİZİ (ARRAY) DÖNÜŞÜMÜ
X=x.values
Y=y.values

###3 LİNEER REGRESYON
from sklearn.linear_model import LinearRegression
lr1=LinearRegression()
lr1.fit(X,Y)


###4 POLYNOMİAL REGRESYON
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
x_poly = pr.fit_transform(X)
lr2=LinearRegression()
lr2.fit(x_poly,y)

#Derece arttıkça daha doğru değerler 
pr3=PolynomialFeatures(degree=4)
x_poly3 = pr3.fit_transform(X)
lr3=LinearRegression()
lr3.fit(x_poly3,y)


###5 GÖRSELLEŞTİRME
plt.scatter(X,Y,color="red")
plt.plot(x,lr1.predict(X),color="blue")
plt.show()

plt.scatter(X,Y,color="red")
plt.plot(x,lr2.predict(x_poly),color="blue")
plt.show()

plt.scatter(X,Y,color="red")
plt.plot(x,lr3.predict(x_poly3),color="blue")
plt.show()


###6 TAHMİN
print("------------------------")
print(lr1.predict(np.array([[11]])))
print(lr1.predict(np.array([[6.6]])))

print("------------------------")
print(lr2.predict(pr.fit_transform(np.array([[11]]))))
print(lr2.predict(pr.fit_transform(np.array([[6.6]]))))

print("------------------------")
print(lr3.predict(pr3.fit_transform(np.array([[11]]))))
print(lr3.predict(pr3.fit_transform(np.array([[6.6]]))))

print("------------------------")