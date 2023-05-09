# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 01:32:44 2023

@author: Aydogn
"""

"""
Öncelikle ön işlemeyle eksik verileri tamamlayacağız ve işe yarar verileri
yeni csv dosyasında toplayacağız.

Madalya kazananlar üzerinde ilerleyeceğiz

"""

##Kütüphane
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter


##Python uyarı kapatma
import warnings
warnings.filterwarnings("ignore")


##Veri
veri=pd.read_csv("athlete_events.csv") 

##Veri hakkında genel bilgi
#veri.info()


##Veri stün ad değiştirme
veri.rename(columns={'ID'     : 'id',
                     'Name'   : 'isim',
                     'Gender' : 'cinsiyet',
                     'Age'    : 'yas',
                     'Height' : 'boy',
                     'Weight' : 'kilo',
                     'Team'   : 'takim',
                     'NOC'    : 'uok',
                     'Games'  : 'oyunlar',
                     'Year'   : 'yil',
                     'Season' : 'sezon',
                     'City'   : 'sehir',
                     'Event'  : 'etkinlik',
                     'Medal'  : 'madalya'}, inplace=True)
    #inplace = True ayrı bir yere yazmak yerine üstüne yazar
#print(veri.head(1)) #head() Return the first n rows.


##Veride stün çıkarma
veri=veri.drop(["id","oyunlar"], axis=1) # axis=1 stünu ifade eder 0 satırı
#print(veri.head(1))

##Eksik veri tamamlama

essiz_etkinlik=pd.unique(veri.etkinlik)
    #uniqe() aynı verileri küme gibi tek değişken yapar
#print(len(essiz_etkinlik)) #765
    
veri_gecici=veri.copy()
boy_kilo_liste=["boy","kilo"]

for e in essiz_etkinlik:
    
    #etkinlik filtresi
    etkinlik_filtre = veri_gecici.etkinlik == e
    
    #veriyi etkinliğe göre filtreleme
    veri_filtreli=veri_gecici[etkinlik_filtre]
    
    #boy ve kilo için etkinlik özelinde ortalama hesaplama
    for s in boy_kilo_liste:
        ortalama = np.round(np.mean(veri_filtreli[s]),2)
        
        if not np.isnan(ortalama): #eğer etkinlik özelinde ortalama varsa
            veri_filtreli[s]=veri_filtreli[s].fillna(ortalama)
        
        else: #eğer etkinlik özelinde ortalama yoksa ortalamayı hesapla
            tum_veri_ortalaması= np.round(np.mean(veri[s]),2)
            veri_filtreli[s]=veri_filtreli[s].fillna(tum_veri_ortalaması)
            
    #etkinlik özelinde kayıp değerleri doldurulmuş olan veriyi, veri_gecici'ye
    # eşitleyelim
    veri_gecici[etkinlik_filtre]= veri_filtreli
            
#kayıp değerleri giderilmiş olan geçici veriyi gerçek veriye eşitleyelim
veri=veri_gecici.copy()
#veri.info() #boy ve kilo sütunlarında kayıp eğer sayısına bakalım             
      
#yas değişkeninde tanımlı olmayan değerleri bulalım      
yas_ortalamasi=np.round(np.mean(veri.yas),2)
print(f"Yaş ortalaması: {yas_ortalamasi}")
veri["yas"]=veri["yas"].fillna(yas_ortalamasi)           
veri.info()           
            
madalya_degiskeni = veri["madalya"]
   
            
madalya_degiskeni_filtresi = ~ pd.isnull(madalya_degiskeni)
veri=veri[madalya_degiskeni_filtresi]   

veri.info()

#veri.to_csv("olimpiyatlar_temizlenmis.csv",index=False) 
#oluşturulan veriler yeni csv dosyasına atılır
