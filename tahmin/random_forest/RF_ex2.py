#Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Veri
data = pd.read_csv("ortopedik_hastaların_biyomekanik_özellikleri.csv")

#kategorik-> numerik
data["class"] = [1 if i=="Abnormal" else 0 for i in data["class"]]

y = data["class"].values 
x_data = data.drop(["class"],axis=1) #drop fonksiyonu verilen satır(axis=0),
# stünu(axis=1) veriden çıkarmaya yarar.

#veriyi normalize etmek
x = (x_data - x_data.min())/(x_data.max() - x_data.min()).values

#eğitim test için bölünmesi
from sklearn.model_selection import train_test_split
#%15 test %85 eğitim
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.15,
                                                    random_state=(42))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
#eğitim
rf = RandomForestClassifier(n_estimators= 100, random_state=1)#100 adet dt
rf.fit(x_train, y_train)
#test
print(f"RF doğruluk: {rf.score(x_test, y_test)}")


# confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
#sıcaklık haritası
f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="white", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()





