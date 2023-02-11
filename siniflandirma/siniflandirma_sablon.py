#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Hata matrisi oluşturma 
from sklearn.metrics import confusion_matrix

# ROC TPR FPR değerleri
from sklearn import metrics

# Accuracy score hasplama
from sklearn.metrics import accuracy_score

#2.veri onisleme
#2.1.veri yukleme
#veriler = pd.read_csv('iris.csv')
veriler = pd.read_excel('iris.xls')
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

# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred_lr = logr.predict(X_test)
y_pro_lr=logr.predict_proba(X_test) #ROC için kullanılır.

cm_lr = confusion_matrix(y_test, y_pred_lr)
print("LR")
print(cm_lr)
#print(y_pred_lr)

acc_lr=accuracy_score(y_test, y_pred_lr)


# 2. KNN
from sklearn.neighbors import KNeighborsClassifier
#n_neighbors komşulara bakaraktan benzerlik sağlayan knn'nin kaç
# komşuya bakması gerektiğine söylüyor arttırılması mantıklı
# olmayabilir bazen.
knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski', weights='distance')
knn.fit(X_train,y_train)

y_pred_knn=knn.predict(X_test)
y_pro_knn=knn.predict_proba(X_test) #ROC için kullanılır.

cm_knn = confusion_matrix(y_test, y_pred_knn)
print("KNN")
print(cm_knn)
#print(y_pred_knn)

acc_knn=accuracy_score(y_test, y_pred_knn)


# 3. SVC (SVM Classifier)
from sklearn.svm import SVC
#kernel 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
svc=SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred_svc=svc.predict(X_test)
#y_pro_svc=svc.predict_proba(self,X_test) #ROC için kullanılır.

cm_svc = confusion_matrix(y_test, y_pred_svc)
print("SVC")
print(cm_svc)
#print(y_pred_svc)

acc_svc=accuracy_score(y_test, y_pred_svc)


# 4. Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train, y_train)

y_pred_nb=gnb.predict(X_test)
y_pro_nb=gnb.predict_proba(X_test) #ROC için kullanılır.

cm_nb = confusion_matrix(y_test, y_pred_nb)
print("GNB")
print(cm_nb)
#print(y_pred_nb)

acc_nb=accuracy_score(y_test, y_pred_nb)


# 5. Decision Tree
from sklearn.tree import DecisionTreeClassifier
#criterion = gini(default) / entropy
dtc=DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)

y_pred_dt=dtc.predict(X_test)
y_pro_dt=dtc.predict_proba(X_test) #ROC için kullanılır.

cm_dt=confusion_matrix(y_test, y_pred_dt)
print("DTC")
print(cm_dt)
#print(y_pred_dt)

acc_dt=accuracy_score(y_test, y_pred_dt)


# 6. Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion = 'gini')
rfc.fit(X_train,y_train)

y_pred_rf=rfc.predict(X_test)
y_pro_rf=rfc.predict_proba(X_test) #ROC için kullanılır.

cm_rf=confusion_matrix(y_test, y_pred_rf)
print("RFC")
print(cm_rf)
#print(y_pred_rf)

acc_rf=accuracy_score(y_test, y_pred_rf)


# ROC TPR FPR değerleri
from sklearn import metrics
fpr, tpr, thold =metrics.roc_curve(y_test, y_pro_knn[:,0],pos_label='e')




