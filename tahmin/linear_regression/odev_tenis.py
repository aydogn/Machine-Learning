#KÜTÜPHANELER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#VERİ
veriler = pd.read_csv('odev_tenis.csv')
outlook = veriler.iloc[:,0:1].values
windy = veriler.iloc[:,3:4].values
play = veriler.iloc[:,4:].values

#encoder nominal ordinal (kategorik) -> numeric
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
outlook[:,0] = le.fit_transform(veriler.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
windy[:,0] = le.fit_transform(veriler.iloc[:,3:4])
ohe = preprocessing.OneHotEncoder()
windy = ohe.fit_transform(windy).toarray()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
play[:,0] = le.fit_transform(veriler.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
play = ohe.fit_transform(play).toarray()

#Kısa encoder metodu 
#veriler2= veriler.apply(le.fit_transform)
#bütün verileri çevirmek uygun olmaz tek tek yapmak en doğrusudur
#bunu uyguladıktan sonra 3 değer içeren değerler 0,1,2 şeklinde
#değerlendirilidiği için hot encoder yapılması gerekir


#Data frame dönüşümü
sonuc1 = pd.DataFrame(data=outlook, index=range(14), columns=["overcast","rainy","sunny"])
sonuc2 = pd.DataFrame(data=windy[:,1:2], index=range(14), columns = ["true"])
sonuc3= pd.DataFrame(data=play[:,1:2], index=range(14), columns=["yes"])
kalan=pd.DataFrame(data=veriler.iloc[:,1:3], index=range(14), columns=["temperature","humidity"])

s1=pd.concat([sonuc1,sonuc2],axis=1)
s2=pd.concat([s1,sonuc3],axis=1)
s3=pd.concat([s2,kalan],axis=1)
#print(s3)


#Verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(s3.iloc[:,:-1],kalan.iloc[:,1],test_size=0.33,random_state=0)


#lineer regresyon
from sklearn.linear_model import LinearRegression
r1 = LinearRegression()
r1.fit(x1_train,y1_train)
y1_pred = r1.predict(x1_test)



#Geri eleme metoduyla daha doğru veri elde etme
import statsmodels.api as sm
X=np.append(arr = np.ones((14,1)).astype(int), values = s3.iloc[:,:-1], axis=1)
#X=sm.add_constant(veri)

humidity_tahmin = s3.iloc[:,-1].values

X_l= s3.iloc[:,[0,1,2,3,4,5]].values
X_l= np.array(X_l,dtype=(float))
model=sm.OLS(humidity_tahmin,X_l).fit()
print(model.summary()) 
#p value en çok 3.indexte var bu bozduğunun göstergesi


X_l= s3.iloc[:,[0,1,2,4,5]].values
X_l= np.array(X_l,dtype=(float))
model=sm.OLS(humidity_tahmin,X_l).fit()
print(model.summary())


#Birleştirme yapıp tekrar eğitime sokma
sol = s3.iloc[:,0:3]
sag = s3.iloc[:,4:-1]
veri_d = pd.concat([sol,sag],axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(veri_d,kalan.iloc[:,1],test_size=0.33,random_state=0)


#Tekrar lineer regresyon
from sklearn.linear_model import LinearRegression
r2 = LinearRegression()
r2.fit(x_train,y_train)
y2_pred = r2.predict(x_test)
#y1_pred'in y2_pred'ten çok daha iyi olduğunu y1_testten anlayabiliyoruz.



