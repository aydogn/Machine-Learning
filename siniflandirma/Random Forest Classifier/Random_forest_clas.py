
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

###Burda eğitim verisiyle eğitim söz konusu akabinde tahmin etme var
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print("LR")
print(y_pred)

###Hata matrisi oluşturma 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)


from sklearn.neighbors import KNeighborsClassifier
#n_neighbors komşulara bakaraktan benzerlik sağlayan knn'nin kaç
# komşuya bakması gerektiğine söylüyor arttırılması mantıklı
# olmayabilir bazen.
knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski', weights='distance')
knn.fit(X_train,y_train)

y_pred2=knn.predict(X_test)

cm2 = confusion_matrix(y_test, y_pred2)
print("KNN")
print(cm2)
print(y_pred2)


from sklearn.svm import SVC
svc=SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred3=svc.predict(X_test)

cm3 = confusion_matrix(y_test, y_pred3)
print("SVC")
print(cm3)
print(y_pred3)


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train, y_train)

y_pred4=gnb.predict(X_test)

cm4 = confusion_matrix(y_test, y_pred4)
print("GNB")
print(cm4)
print(y_pred4)


from sklearn.tree import DecisionTreeClassifier
#criterion = gini(default) / entropy
dtc=DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)

y_pred5=dtc.predict(X_test)

cm5=confusion_matrix(y_test, y_pred5)
print("DTC")
print(cm5)
print(y_pred5)


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion = 'gini')
rfc.fit(X_train,y_train)

y_pred6=rfc.predict(X_test)

cm6=confusion_matrix(y_test, y_pred6)
print("RFC")
print(cm6)
print(y_pred6)







