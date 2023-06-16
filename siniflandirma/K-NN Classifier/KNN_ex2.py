#Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#uyarı kapatma
import warnings 
warnings.filterwarnings("ignore")


#Veri
data = pd.read_csv("ortopedik_hastaların_biyomekanik_özellikleri.csv")

#Görselleştirme
sns.scatterplot(data=data, x="lumbar_lordosis_angle", y="pelvic_tilt numeric",
                hue="class")
plt.xlabel("lomber lordoz açısı")
plt.ylabel("pelvik eğim")
plt.legend()
plt.show()

#kategorik-> numerik
data["class"] = [1 if i=="Abnormal" else 0 for i in data["class"]]

y = data["class"].values 
x_data = data.drop(["class"],axis=1) #drop fonksiyonu verilen satır(axis=0),
# stünu(axis=1) veriden çıkarmaya yarar.

#veriyi normalize etmek
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values

#eğitim test için bölünmesi
from sklearn.model_selection import train_test_split
#%15 test %85 eğitim
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.15,
                                                    random_state=(42))

from sklearn.neighbors import KNeighborsClassifier
komsu_sayisi = 3
knn = KNeighborsClassifier(n_neighbors=komsu_sayisi)
knn.fit(x_train,y_train)

skor=knn.score(x_test,y_test)
prediction = knn.predict(x_test)

# print("{} komşulu model doğruluk oranı: {}".format(komsu_sayisi,skor))

#En iyi K değirinin bulunması
score_list = [0]
max=score_list[0]
for i in range(1,50):
    knn2=KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    if max < score_list[i]:
        max=score_list[i]

array = np.array(score_list)
array = array[array>0]    
plt.plot(range(1,50), array)
plt.xlabel("k değerleri")
plt.ylabel("doğruluk")
plt.title("En iyi K değirinin bulunması")
plt.show()
print(f"En uygun K skoru: {max}")


