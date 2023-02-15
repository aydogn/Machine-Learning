#KÜTÜPHANELER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#VERİ
veriler = pd.read_csv('cinsiyet_boy_kilo_yas.csv')
x = veriler.iloc[:,1:4]
y = veriler.iloc[:,4:]
X=x.values
Y=y.values


#DataFrame
sonuc= pd.DataFrame(data=Y, index=range(22), columns=["cinsiyet"])
s=pd.concat([veriler.iloc[:,1:4],sonuc],axis=1)


#Numeric
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
Y[:,0] = le.fit_transform(veriler.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
Y = ohe.fit_transform(Y).toarray()
Y=Y[:,0]

#Sadece corr fonksiyonuna bakaraktan hangi bağımlı değişkenler hangi bağımsız  
# değişkenlere nasıl etki ediyor öğrenebiliriz.
print(s.corr())


###TAHMİN (MODELLER)

#Hangi sütunların gerekli olduğunu belirlemek için
# P value larını hesaplamak gerekir
import statsmodels.api as sm

#LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lr1=LinearRegression()
lr1.fit(X,Y)

print("linear OLS")
model1 = sm.OLS(lr1.predict(X),X)
print(model1.fit().summary())


#POLYNOMIAL REGRESSION
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


#SUPPORT VECTOR REGRESSION
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


#DECISION TREE
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

print("DECISION TREE OLS")
model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())


#RANDOM FOREST REGRESSION
from sklearn.ensemble import RandomForestRegressor
# n_estimators -> kaç tane decision tree kullanacağını verir.
# En doğru sonuç için arttırmak çözüm değildir arttırınca değerler 
#sabitleşir.
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())

print("RF OLS")
model5 = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())


#LOGISTIC REGRESSION
#Daha çok sınıflandırmada kullanıldığı için açıklamayı sınıflandırma bölümünde
# yapacağım.
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X,Y)

print("LOGISTIC REGRESSION OLS")
model6 = sm.OLS(logr.predict(X),X)
print(model4.fit().summary())


##R2 KIYASLAMA
# r2'nin 1'e yaklaşması doğruluğunu arttırır ama decision tree gibi
#tahminden ziyade ezbere dayalı regressyonların tuzağına düşer.
from sklearn.metrics import r2_score

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

print("------------------------")
print("LOGISTIC REGRESSION")
print("r2:",r2_score(Y,logr.predict(X)))
print("-"*50)

liste=[168,69,20]
print("LINEAR REGRESSION",lr1.predict(np.array([liste])))

print("POLYNOMIAL REGRESSION 2 degree",lr2.predict(pr.fit_transform(np.array([liste]))))

print("POLYNOMIAL REGRESSION 4 degree",lr3.predict(pr3.fit_transform(np.array([liste]))))

print("SUPPORT VECTOR REGRESSION",svr_reg.predict(np.array([liste])))

print("DECISION TREE",r_dt.predict([liste]))

print("RANDOM FOREST REGRESSION",rf_reg.predict([liste]))

print("LOGISTIC REGRESSION",logr.predict([liste]))



