#1.KÜTÜPHANELER
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



###2.VERİ ÖN İŞLEME 
###########################################################################

#Bu dosyada olmayan verileri diğer verilerin ortalamasını alarak 
#yerleştirdik. Ancak tabii ki farklı ülkelerin verileri diğer ülkeleri 
#etkilemeyebilir. Bu yüzden her ülkenin verisinin ortalamasını alarak 
#o ülkenin bilinmeyenine aktardık.



#Eksik değerleri, her sütun boyunca açıklayıcı bir istatistik 
#(örneğin, ortalama, medyan veya en sık) kullanarak veya sabit bir 
#değer kullanarak değiştirin.

#from sklearn.impute import SimpleImputer

#veriler = pd.read_csv('eksikveriler.csv')

#missing_values=Eksik değerler için yer tutucu
#strategy =  “mean” “median” “most_frequent” “constant” değerlerini alır
#imputer = SimpleImputer(missing_values = np.nan , strategy="mean")

#Yas1 = veriler.iloc[9:15,1:4].values
#Yas2 = veriler.iloc[15:,1:4].values


#fit() arkaplanda işlem gerçekleştirir ancak göstermez yazmaz işlemez
#imputer1 = imputer.fit(Yas1[:,:])

#transform() gerçekleştirilen işlemi göstermeye yarar
#Yas1[:,:] = imputer1.transform(Yas1[:,:])
#print(Yas1)

#print('-----------------------------------------------')

#imputer2 = imputer.fit(Yas2[:,:])
#Yas2[:,:] = imputer2.transform(Yas2[:,:])
#print(Yas2)"""

#########################

#Eğer ki bütün değerlerin ortalamasını alarak eksikleri tamamlamak
#istersek alttak yazılım devreye girmeli.

###2.1 Veri Yükleme
veriler = pd.read_csv('eksikveriler.csv')

###2.2 eksik veriler
from sklearn.impute import SimpleImputer
all_imputer = SimpleImputer(missing_values = np.nan , strategy="mean")

Yas = veriler.iloc[:,1:4].values

#fit() arkaplanda işlem gerçekleştirir ancak göstermez yazmaz işlemez
#transform() gerçekleştirilen işlemi göstermeye yarar

Yas[:,1:4]= all_imputer.fit_transform(Yas[:,1:4])
print(Yas)
###########################################################################
###2.3 encoder nominal ordinal (kategorik) -> numeric
from sklearn.impute import SimpleImputer
veriler = pd.read_csv('eksikveriler.csv')

ulke = veriler.iloc[:,0:1].values

#print(ulke)

from sklearn import preprocessing

#Bu işlemle LabelEncoder() fit_transformla birlikte önce verileri alfabetik
#şekilde sıralıyoruz sonra her birine sırasına göre 0 1 2 .. veriyoruz
#fr:0 tr:1 us:2 aldı.
le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
#print(ulke)

#OneHotEncoder()'la bizler önceki aldığımız ve sayılara çevirdiğimiz
#verileri 1 ve 0 lara çeviren bir fonksiyon bunu yapmak için de kaç tane
#veri varsa o kadar sutün oluşturuyor ve o sutünlara varsa 1 yoksa 0 
#yazıyor.
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

##########################################################################
###2.4 Data frame dönüşümü

#DataFrame ile taplo oluşturuyoruz
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=["fr","tr","us"])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index=range(22), columns = ["boy","kilo","yas"])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
sonuc3= pd.DataFrame(data=cinsiyet, index=range(22), columns=["cinsiyet"])
print(sonuc3)

#Burda artık elde ettiğimiz ve işlediğimiz verileri birleştiriyoruz
#axis=1 ilk dataframe ile ikinci dataframe in kolonlarını birleştiriyor
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

#########################################################################
###3 Verilerin eğitim ve test için bölünmesi

from sklearn.model_selection import train_test_split

#s'yi x'e atadı sonuc3'e de y'ye test_size'da 1/3 test için kalanı train
#maksat train ve test veri kümelerine ayırmak 
#dikey eksende bağımlı bağımsız, yatay eksende train ve test ayrımı yapılır
x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)
#Train, eğitim kısmıdır en uygun algoritma seçilir.
#Validation, Train in içindedir Train ile seçilen model iyileştirilir.
#x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.25,shuffle=False,random_state=0)
#Shuffle, satırları dikey yönünde karıştırır.

##########################################################################
###4 verilerin ölçeklendirilmesi

from sklearn.preprocessing import StandardScaler

#Ortalama değer 0, standart sapma 1 yapar. Değerler yakınlaştırılır.
#Elimizdeki değerden ortalama değeri çıkartıyoruz, sonrasında 
#varyans değerine bölüyoruz.
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

##########################################################################
###5 Tahmin

#Burda eğitim verisiyle eğitim söz konusu akabinde tahmin etme var
#logr.fit(bağımsız,bağımlı)
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)

print(y_pred)
print(y_test)

##########################################################################

#ilk listenin ilk elemanı ikinci listenin ikinci elemanı doğru sayısını
#diğerleri yanlış sayısını gösterir
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

#https://www.veribilimiokulu.com/lojistik-regresyon-logistic-regression-classification-ile-siniflandirma-python-ornek-uygulamasi/
#üstteki sitede örnek bir yazılım var bu konuyla ilgili
