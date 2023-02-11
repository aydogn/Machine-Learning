#KÜTÜPHANELER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###Veri 
veriler = pd.read_csv('eksikveriler.csv')
ulke = veriler.iloc[:,0:1].values
Yas = veriler.iloc[:,1:4].values
cinsiyet = veriler.iloc[:,-1].values


###eksik veriler
from sklearn.impute import SimpleImputer
all_imputer = SimpleImputer(missing_values = np.nan , strategy="mean")
Yas[:,1:4]= all_imputer.fit_transform(Yas[:,1:4])

###encoder nominal ordinal (kategorik) -> numeric
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()


###Data frame dönüşümü
#DataFrame ile tablo oluşturuyoruz
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=["fr","tr","us"])
sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns = ["boy","kilo","yas"])
sonuc3= pd.DataFrame(data=cinsiyet, index=range(22), columns=["cinsiyet"])
s=pd.concat([sonuc,sonuc2],axis=1)
s2=pd.concat([s,sonuc3],axis=1)


###Verilerin eğitim ve test için bölünmesi

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)

###verilerin ölçeklendirilmesi

from sklearn.preprocessing import StandardScaler
#Ortalama değer 0, standart sapma 1 yapar. Değerler yakınlaştırılır.
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

###Tahmin

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train) #logr.fit(bağımsız,bağımlı)

y_pred = logr.predict(X_test)


#hata matriksi
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
