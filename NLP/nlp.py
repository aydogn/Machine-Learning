"""
Created on Mon Jan 30 17:37:30 2023

@author: abdullahaydogan
"""

import numpy as np
import pandas as pd


with open("Restaurant_Reviews.csv","r") as file:

    reviews = []

    liked = []

    for i, line in enumerate(file):

        if i == 0:

            continue

        while not line[-1].isdigit():

            line = line[:-1]

            if line[-1].isdigit():

                    break

        reviews.append(line[:-2])

        liked.append(line[-1])

yorumlar = pd.DataFrame(data=zip(reviews, liked), columns=['Review', 'Liked'])

import re
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

aritilmis_liste=[]

for i in range(1000):
    yorum = re.sub('[^a-zA-Z]', ' ', yorumlar['Review'][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum 
             if not kelime in set(stopwords.words('english'))] 
    yorum = ' '.join(yorum)
    aritilmis_liste.append(yorum)


### FEATURE EXTRACTION
### BAG OF WORDS (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=(2000))
X = cv.fit_transform(aritilmis_liste).toarray() # bağımsız değişken
Y = yorumlar.iloc[:,1].values # bağımlı değişken


### MACHINE LEARNING
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state= 0)

from sklearn.neighbors import KNeighborsClassifier
#n_neighbors komşulara bakaraktan benzerlik sağlayan knn'nin kaç
# komşuya bakması gerektiğine söylüyor arttırılması mantıklı
# olmayabilir bazen.
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', weights='distance')
knn.fit(X_train,Y_train)

y_pred2=knn.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred2)
print(cm)


