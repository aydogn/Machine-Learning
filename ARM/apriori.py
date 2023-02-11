# Apriori


#Tavsiye edilen algoritma budur büyük verilerde kullanılır.
# Breadth first search kullnılır.
#Eclat algoritması da bunun bir uzantısıdır.Farkı Depth first search kullanır.
# Kısa veriler için uygundur hızlı çalışır ama apriori en uygun kullanımdır.
 
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('sepet.csv', header=None)

liste = []

for i in range(0,7501):
    liste.append([str(veriler.values[i,j]) for j in range(0,20)])

#apyori kütüphanesini github'tan indirdik, sklearn'de her şey yok tabi :)
from apyori import apriori
kurallar = apriori(liste,min_support=0.01, min_confidence=0.2, min_lift = 3, 
                   min_length=2)

#generator object olduğu için (kurallar), listeye çevirmek gerekir. 
#lift değeri, ilk verilen ürünü/leri alınca normalden x kadar artış gösteriyor
# manasına gelir
print(list(kurallar))


