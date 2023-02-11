###1 KÜTÜPHANELER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# r2'nin 1'e yaklaşması doğruluğunu arttırır ama decision tree gibi
#tahminden ziyade ezbere dayalı regressyonların tuzağına düşer.
from sklearn.metrics import r2_score

###2 VERİ YÜKLEME
veriler = pd.read_csv('Maaslar.csv')

#DATAFRAME DİLİMLEME(SLICE)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#NUMPY DİZİ (ARRAY) DÖNÜŞÜMÜ
X=x.values
Y=y.values

###3 TAHMİN (MODELLER)
"""
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


#3.5 RANDOM FOREST REGRESSION
from sklearn.ensemble import RandomForestRegressor
# n_estimators -> kaç tane decision tree kullanacağını verir.
# En doğru sonuç için arttırmak çözüm değildir arttırınca değerler 
#sabitleşir.
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())


###4 GÖRSELLEŞTİRME, TAHMİNİ SAYISAL DEĞERLER VE R2
plt.scatter(X,Y,color="red")
plt.plot(x,lr1.predict(X),color="blue")
plt.show()
print("------------------------")
print("LINEAR REGRESSION")
print(lr1.predict(np.array([[11]])))
print(lr1.predict(np.array([[6.6]])))
print("------------------------")
print("r2:",r2_score(Y,lr1.predict(X)))


plt.scatter(X,Y,color="red")
plt.plot(x,lr2.predict(x_poly),color="blue")
plt.show()
plt.scatter(X,Y,color="red")
plt.plot(x,lr3.predict(x_poly3),color="blue")
plt.show()
print("------------------------")
print("POLYNOMIAL REGRESSION")
print(lr2.predict(pr.fit_transform(np.array([[11]]))))
print(lr2.predict(pr.fit_transform(np.array([[6.6]]))))
print("------------------------")
print("r2:",r2_score(Y,lr2.predict(x_poly)))
print("r2:",r2_score(Y,lr3.predict(x_poly3)))


plt.scatter(x_olcekli,y_olcekli,color="red")
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color="blue")
plt.show()
print("------------------------")
print("SUPPORT VECTOR REGRESSION")
print(lr3.predict(pr3.fit_transform(np.array([[11]]))))
print(lr3.predict(pr3.fit_transform(np.array([[6.6]]))))
print(svr_reg.predict(np.array([[11]])))
print(svr_reg.predict(np.array([[6.6]])))
#Standard Scalerdan normal değere çevirme
#print(sc2.inverse_transform(svr_reg.predict(sc1.transform(np.array([[11]]).reshape(1,-1)))))
print("------------------------")
print("r2:",r2_score(y_olcekli,svr_reg.predict(x_olcekli)))


plt.scatter(X,Y,color="red")
plt.plot(X,r_dt.predict(X),color="blue")
plt.show()
print("------------------------")
print("DECISION TREE")
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))
print("------------------------")
print("r2:",r2_score(Y,r_dt.predict(X)))


plt.scatter(X,Y,color="red")
plt.plot(X,rf_reg.predict(X),color="blue")
plt.show()
print("------------------------")
print("RANDOM FOREST REGRESSION")
print(rf_reg.predict([[11]]))
print(rf_reg.predict([[6.6]]))
print("------------------------")
print("r2:",r2_score(Y,rf_reg.predict(X)))

##R2 KIYASLAMA
print("R2 ÖZET")
print("------------------------")
print("LINEAR REGRESSION")
print("r2:",r2_score(Y,lr1.predict(X)))

print("------------------------")
print("POLYNOMIAL REGRESSION")
print("r2:",r2_score(Y,lr2.predict(x_poly)))
print("r2:",r2_score(Y,lr3.predict(x_poly3)))

print("------------------------")
print("SUPPORT VECTOR REGRESSION")
print("r2:",r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

print("------------------------")
print("DECISION TREE")
print("r2:",r2_score(Y,r_dt.predict(X)))

print("------------------------")
print("RANDOM FOREST REGRESSION")
print("r2:",r2_score(Y,rf_reg.predict(X)))
"""
#KOYALAMA İÇİN KOLAYLIK
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


#3.1 LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lr1=LinearRegression()
lr1.fit(X,Y)

plt.scatter(X,Y,color="red")
plt.plot(x,lr1.predict(X),color="blue")
plt.show()

print("------------------------")
print("LINEAR REGRESSION")
print(lr1.predict(np.array([[11]])))
print(lr1.predict(np.array([[6.6]])))
print("------------------------")
print("r2:",r2_score(Y,lr1.predict(X)))


#3.2 POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
x_poly = pr.fit_transform(X)
lr2=LinearRegression()
lr2.fit(x_poly,y)

pr3=PolynomialFeatures(degree=4)
x_poly3 = pr3.fit_transform(X)
lr3=LinearRegression()
lr3.fit(x_poly3,y)

plt.scatter(X,Y,color="red")
plt.plot(x,lr2.predict(x_poly),color="blue")
plt.show()

plt.scatter(X,Y,color="red")
plt.plot(x,lr3.predict(x_poly3),color="blue")
plt.show()

print("------------------------")
print("POLYNOMIAL REGRESSION")
print(lr2.predict(pr.fit_transform(np.array([[11]]))))
print(lr2.predict(pr.fit_transform(np.array([[6.6]]))))
print("------------------------")
print("r2:",r2_score(Y,lr2.predict(x_poly)))
print("r2:",r2_score(Y,lr3.predict(x_poly3)))


#3.3 SUPPORT VECTOR REGRESSION
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))
from sklearn.svm import SVR
svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color="red")
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color="blue")
plt.show()

print("------------------------")
print("SUPPORT VECTOR REGRESSION")
print(lr3.predict(pr3.fit_transform(np.array([[11]]))))
print(lr3.predict(pr3.fit_transform(np.array([[6.6]]))))
print(svr_reg.predict(np.array([[11]])))
print(svr_reg.predict(np.array([[6.6]])))
#Standard Scalerdan normal değere çevirme
#print(sc2.inverse_transform(svr_reg.predict(sc1.transform(np.array([[11]]).reshape(1,-1)))))
print("------------------------")
print("r2:",r2_score(y_olcekli,svr_reg.predict(x_olcekli)))


#3.4 DECISION TREE
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

plt.scatter(X,Y,color="red")
plt.plot(X,r_dt.predict(X),color="blue")
plt.show()

print("------------------------")
print("DECISION TREE")
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))
print("------------------------")
print("r2:",r2_score(Y,r_dt.predict(X)))


#3.5 RANDOM FOREST REGRESSION
from sklearn.ensemble import RandomForestRegressor
# n_estimators -> kaç tane decision tree kullanacağını verir.
# En doğru sonuç için arttırmak çözüm değildir arttırınca değerler 
#sabitleşir.
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())

plt.scatter(X,Y,color="red")
plt.plot(X,rf_reg.predict(X),color="blue")
plt.show()

print("------------------------")
print("RANDOM FOREST REGRESSION")
print(rf_reg.predict([[11]]))
print(rf_reg.predict([[6.6]]))
print("------------------------")
print("r2:",r2_score(Y,rf_reg.predict(X)))
