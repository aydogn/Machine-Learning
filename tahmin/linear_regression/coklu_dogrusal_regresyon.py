#KÜTÜPHANELER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#VERİ
veriler = pd.read_csv('veriler.csv')
ulke = veriler.iloc[:,0:1].values
kalan = veriler.iloc[:,1:4].values
c = veriler.iloc[:,-1:].values
cinsiyet = veriler.iloc[:,-1].values


#nominal->numeric
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()


le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(veriler.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()


#Data frame dönüşümü
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=["fr","tr","us"])
sonuc2 = pd.DataFrame(data=kalan, index=range(22), columns = ["boy","kilo","yas"])
sonuc3= pd.DataFrame(data=c[:,:1], index=range(22), columns=["cinsiyet"])

s=pd.concat([sonuc,sonuc2],axis=1)
s2=pd.concat([s,sonuc3],axis=1)


#Verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)



#Çoklu lineer regresyon birden fazla değişkene bağlı olan ve bağımlı
# değişkeni doğrusal bir artış gösteren verisetlerindeki değişkenlerin
# arasındaki bağıntıyı bulmaya yarayan yöntemdir.

###Lineer Regresyon (çoklu)
from sklearn.linear_model import LinearRegression
r1 = LinearRegression()
r1.fit(x1_train,y1_train)
y1_pred = r1.predict(x1_test)


boy = s2.iloc[:,3:4].values
sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]
veri = pd.concat([sol,sag],axis=1)

#boy tahmini için eğitim ve test
x2_train, x2_test, y2_train, y2_test = train_test_split(veri,boy,test_size=0.33,random_state=0)

#boy tahmini için Lineer Regresyon (çoklu)
r2 = LinearRegression()
r2.fit(x2_train,y2_train)
y2_pred = r2.predict(x2_test)


#OLS sonuçları 
import statsmodels.api as sm
#buna bakarak hangi stünlar öğrenimi zorlaştırıyor anlayabiliriz

# Çoklu regresyonda (Y=B0 +B1X1 +B2X2 ... +e) B0 ı eklememiz lazım 
# ona da "1" ekleyeceğiz, veri değişkenine aktaracağız
# axis default değeri satıra ekle demek, 1 olursa stün ekle demek
X=np.append(arr = np.ones((22,1)).astype(int), values = veri, axis=1)
#X=sm.add_constant(veri)
X_l= veri.iloc[:,[0,1,2,3,4,5]].values
X_l1= np.array(X_l,dtype=(float))
model=sm.OLS(boy,X_l1).fit()
print(model.summary())


