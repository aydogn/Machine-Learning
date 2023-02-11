# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 23:51:56 2023

@author: abdullahaydogan
"""

import numpy as np
import pandas as pd

veriler = pd.read_csv('Churn_Modelling.csv')

#veri ön işleme
X = veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values

#encoder kategorik -> numeric
from sklearn import preprocessing

le1 = preprocessing.LabelEncoder()
X[:,1] = le1.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])  


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough")

X = ohe.fit_transform(X)
X = X[:,1:]


# verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.33,
                                                     random_state=0)


# verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


#Yapay Sinir Ağı

import keras

from keras.models import Sequential
from keras.layers import Dense

classi = Sequential()

classi.add(Dense(6, kernel_initializer="uniform", activation="relu",
                 input_dim =11))

classi.add(Dense(6, kernel_initializer="uniform", activation="relu"))

classi.add(Dense(1, kernel_initializer="uniform", activation="sigmoid"))

classi.compile(optimizer = 'adam', loss = 'binary_crossentropy',
               metrics = ['accuracy'])


classi.fit(X_train, y_train, epochs=60)

y_pred = classi.predict(X_test)
y_pred = (y_pred>0.5) #true false döndürmek için

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)








