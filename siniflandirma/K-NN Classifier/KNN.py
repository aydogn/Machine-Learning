
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
        
x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)



from sklearn.neighbors import KNeighborsClassifier
#n_neighbors komşulara bakaraktan benzerlik sağlayan knn'nin kaç
# komşuya bakması gerektiğine söylüyor arttırılması mantıklı
# olmayabilir bazen.
knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski', weights='distance')
knn.fit(X_train,y_train)

y_pred2=knn.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)
print(cm)
print(y_pred2)







