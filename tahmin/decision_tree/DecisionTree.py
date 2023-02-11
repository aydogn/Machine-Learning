###1 KÜTÜPHANELER
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

###3 TAHMİN (MODELLER)

#3.1 LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lr1=LinearRegression()
lr1.fit(X,Y)


#3.2 POLYNOMIAL REGRESSION
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


#3.3 SUPPORT VECTOR REGRESSION
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

from sklearn.svm import SVR
svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)


#3.4 DECISION TREE
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)


###4 GÖRSELLEŞTİRME
plt.scatter(X,Y,color="red")
plt.plot(x,lr1.predict(X),color="blue")
plt.show()

plt.scatter(X,Y,color="red")
plt.plot(x,lr2.predict(x_poly),color="blue")
plt.show()

plt.scatter(X,Y,color="red")
plt.plot(x,lr3.predict(x_poly3),color="blue")
plt.show()

plt.scatter(x_olcekli,y_olcekli,color="red")
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color="blue")
plt.show()

plt.scatter(X,Y,color="red")
plt.plot(X,r_dt.predict(X),color="blue")
plt.show()


###5 TAHMİNİ SAYISAL DEĞERLER
print("------------------------")
print("LINEAR REGRESSION")
print(lr1.predict(np.array([[11]])))
print(lr1.predict(np.array([[6.6]])))

print("------------------------")
print("POLYNOMIAL REGRESSION")
print(lr2.predict(pr.fit_transform(np.array([[11]]))))
print(lr2.predict(pr.fit_transform(np.array([[6.6]]))))

print("------------------------")
print("SUPPORT VECTOR REGRESSION")
print(lr3.predict(pr3.fit_transform(np.array([[11]]))))
print(lr3.predict(pr3.fit_transform(np.array([[6.6]]))))

print(svr_reg.predict(np.array([[11]])))
print(svr_reg.predict(np.array([[6.6]])))
#Standard Scalerdan normal değere çevirme
#print(sc2.inverse_transform(svr_reg.predict(sc1.transform(np.array([[11]]).reshape(1,-1)))))

print("------------------------")
print("DECISION TREE")
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))
print("------------------------")