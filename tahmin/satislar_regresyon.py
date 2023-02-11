###1 KÜTÜPHANELER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# r2'nin 1'e yaklaşması doğruluğunu arttırır ama decision tree gibi
#tahminden ziyade ezbere dayalı regressyonların tuzağına düşer.
from sklearn.metrics import r2_score

#Hangi sütunların gerekli olduğunu belirlemek için
# P value larını hesaplamak gerekir
import statsmodels.api as sm

###2 VERİ YÜKLEME
veriler = pd.read_csv('satislar.csv')

#Bağımlı değişkene bağlı olarak en uygun bağımlı 
#değişkeni belirlemede kullanılır.
#Buna OLS üzerinden Rsquare ve p-value ları ile de bakılabilinir.
print(veriler.corr())
#DATAFRAME DİLİMLEME(SLICE)
x = veriler.iloc[:,:1]
y = veriler.iloc[:,1:]

#NUMPY DİZİ (ARRAY) DÖNÜŞÜMÜ
X=x.values
Y=y.values

###3 TAHMİN (MODELLER)

#3.1 LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lr1=LinearRegression()
lr1.fit(X,Y)

print("linear OLS")
model1 = sm.OLS(lr1.predict(X),X)
print(model1.fit().summary())
#burda 2. indexli sütun en iyi sonucu veriyor




#3.2 POLYNOMIAL REGRESSION
from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
x_poly = pr.fit_transform(X)
lr2=LinearRegression()
lr2.fit(x_poly,y)

print("Poly OLS 2 degree")
model21 = sm.OLS(lr2.predict(pr.fit_transform(X)),X)
print(model21.fit().summary())

pr3=PolynomialFeatures(degree=4)
x_poly3 = pr3.fit_transform(X)
lr3=LinearRegression()
lr3.fit(x_poly3,y)

print("Poly OLS 4 degree")
model22 = sm.OLS(lr3.predict(pr3.fit_transform(X)),X)
print(model22.fit().summary())


#3.3 SUPPORT VECTOR REGRESSION
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))
from sklearn.svm import SVR
svr_reg = SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)

print("SVR OLS")
model3 = sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())


#3.4 DECISION TREE
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

print("DECISION TREE OLS")
model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())


#3.5 RANDOM FOREST REGRESSION
from sklearn.ensemble import RandomForestRegressor
# n_estimators -> kaç tane decision tree kullanacağını verir.
# En doğru sonuç için arttırmak çözüm değildir arttırınca değerler 
#sabitleşir.
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())

print("RF OLS")
model5 = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())


##R2 KIYASLAMA
print("R2 ÖZET ")
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

print(lr1.predict(np.array([[11]])))
print(lr2.predict(pr.fit_transform(np.array([[11]]))))
print(lr3.predict(pr3.fit_transform(np.array([[11]]))))
print(r_dt.predict([[11]]))
print(rf_reg.predict([[11]]))
