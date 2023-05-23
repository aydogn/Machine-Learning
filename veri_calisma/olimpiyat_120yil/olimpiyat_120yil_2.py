# -*- coding: utf-8 -*-
"""
@author: Aydogn
"""
"""
    Madalya alanlar arasındaki veri yorumları için veri görselleştirme
"""

##Kütüphane
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter


##Python uyarı kapatma
import warnings
warnings.filterwarnings("ignore")


#Veri
veri=pd.read_csv("olimpiyatlar_temizlenmis.csv")


#Veri hakkında bilgi fonksiyonu
#print(veri.describe())
"""   
               yas           boy          kilo           yil
count  39783.000000  39783.000000  39783.000000  39783.000000
mean      25.918456    177.336690     73.738320   1973.943845
std        5.859569     10.170124     13.979041     33.822857
min       10.000000    136.000000     28.000000   1896.000000
25%       22.000000    170.000000     64.000000   1952.000000
50%       25.000000    177.480000     73.000000   1984.000000
75%       29.000000    184.000000     82.000000   2002.000000
max       73.000000    223.000000    182.000000   2016.000000
"""


#Veri Görselliştirme


#Histogram
"""
def plotHistogram(degisken):
    """ """
        Girdi: değişken/stün ismi

        Çıktı: Histogram
  
    """"""
    plt.figure()
    plt.hist(veri[degisken],bins=85,color="orange")
    #bins verilerin sıklığını ifade eder 
    
    plt.xlabel(degisken)
    plt.ylabel("Frekans")
    plt.title(f"Veri Sıklığı -{degisken}")
    plt.show()
    
sayisal_degisken = ["yas","boy","kilo","yil"]
for i in sayisal_degisken:
    plotHistogram(i)  
"""   
    

#Kutu Grafiği
"""
#Tek veri seti için uygun
plt.boxplot(veri.yas)
plt.title("Yaş Değişkeni İçin Kutu Grafiği")  
plt.xlabel("Deger")
plt.ylabel("yas")
plt.show()
""" 
 

#Çizgi Grafiği
"""   
def plotBar(degisken, n=5):
    """"""
        Girdi: değişken/stün ismi
               n= Gösterilecek eşsiz değer sayısı

        Çıktı: Çubuk Grafiği
    """"""
    veri_ = veri[degisken]
    veri_sayma = veri_.value_counts() #değerlerin sayısını verir
    veri_sayma = veri_sayma[:n]
    plt.figure()
    plt.bar(veri_sayma.index, veri_sayma, color="orange")
    plt.xticks(veri_sayma.index, veri_sayma.index.values)#etiket seçme
    plt.xticks(rotation=45) #45 derecelik açıyla yazsın
    plt.ylabel("Frekans")
    plt.title(f"Veri Sıklığı - {degisken}")
    plt.show()
    print(f"{degisken}: \n {veri_sayma}")
    
kategorik_degisken=["isim","cinsiyet","takim","uok","sezon","sehir","spor",
                    "etkinlik","madalya"]

for i in kategorik_degisken:
    plotBar(i)
"""   

###İKİ DEĞİŞKENLİ VERİ ANALİZİ    

#Dağılım (saçılım) grafiği
"""
erkek=veri[veri.cinsiyet == "M"]
kadın=veri[veri.cinsiyet == "F"]
 
plt.figure()
plt.scatter(erkek.boy,erkek.kilo, alpha=0.4, label="Erkek", color="blue")   
plt.scatter(kadın.boy,kadın.kilo, alpha=0.4, label="Kadın", color="orange")
plt.xlabel("Boy")   
plt.xlabel("Kilo")
plt.title("Boy ve Kilo Arasındaki İlişki")
plt.legend()#Grafik elemanları adlandırdığımız etiketleri grafikte gösterir.
plt.show()    
"""


#korelasyon tablosu (verilerin arasındaki bağlantıyı verir) 
#print(veri.iloc[:,["yas","boy","kilo"]].corr()) 
 
"""
#Kategorik->numeric get_dummies fonksiyonununu kullanarak   
veri_gecici = veri.copy()
veri_gecici = pd.get_dummies(veri_gecici,columns=['madalya']) 
#print(veri_gecici.head(2))    
"""
"""
#takim stünu altında madalya verilerinin toplamı (altın madalyaya göre sıralı)
print(veri_gecici[["takim","madalya_Bronze",
             "madalya_Silver",
             "madalya_Gold"]        #groupby fonks. ile veriler ilgili stün
             ].groupby(["takim"]     # altında kümelenir ve sum() ile toplanır    
                ,as_index=False).sum().sort_values(by="madalya_Gold"
                                                ,ascending=False)[:3])
    
#              takim  madalya_Bronze  madalya_Silver  madalya_Gold
# 462  United States          1233.0          1512.0        2474.0
# 403   Soviet Union           677.0           716.0        1058.0
# 165        Germany           678.0           627.0         679.0  
"""    
      
###ÇOK DEĞİŞKENLİ VERİ ANALİZİ 


#Pivot Tablosu
"""
veri_pivot=veri.pivot_table(index="madalya",
                            columns="cinsiyet",
                            values=["boy","kilo","yas"],
                            aggfunc={"boy":np.mean,
                                     "kilo":np.mean,
                                     "yas":[min, max, np.std, np.mean]} )
    
print(veri_pivot.head(2))  
"""   
  
 
#Anomali Tespiti
"""
def anomaliTespiti(df, ozellik):
    outlier_indices = []
    
    for c in ozellik:
        #1. çeyrek
        Q1 = np.percentile(df[c], 25)#yüzdelik belirten fonksiyon
        #3. çeyrek
        Q3 = np.percentile(df[c], 75)
        #IQR = Inter Quartile Range
        IQR = Q3 - Q1
        #aykırı değer için ek adım miktarı
        outlier_step=1.5*IQR
        #aykırı değeri ve de bulunduğu indeksi tespit edelim
        outlier_list_col =df[(df[c] < Q1 - outlier_step) 
                             | (df[c] > Q3 + outlier_step)].index
        #tespit ettiğimiz indexleri depolayalım
        outlier_indices.extend(outlier_list_col)
        
    #eşsiz aykırı değerleri bulalım
    outlier_indices = Counter(outlier_indices)
    #eğer bir örnek v adet stünda farklı değer ise bunu aykırı kabul edelim
    multiple_outliers = list(i for i,v in outlier_indices.items() if v>1)

    return multiple_outliers

veri_anomali = veri.loc[anomaliTespiti(veri,["yas","kilo","boy"])]
print(veri_anomali.spor.value_counts()) 

    
#    Basketball        64
#    Gymnastics        34
#    Handball           6
#    Athletics          5
#    Sailing            3
#    Diving             3
#    Shooting           1
#    Figure Skating     1
#    Wrestling          1
#    Name: spor, dtype: int64        

#Çubuk grafiği       
plt.figure()
plt.bar(veri_anomali.spor.value_counts().index,
        veri_anomali.spor.value_counts().values)  
plt.xticks(rotation=30)
plt.title("Anomaliye Rastlanan Spor Branşları")    
plt.ylabel("Frekans")
plt.grid(True, alpha = 0.5)
plt.show()
 """      
        
veri_zaman = veri.copy()
essiz_yillar = veri_zaman.yil.unique()       
 
dizili_array = np.sort(essiz_yillar) #olimpiyat yapılan yılları sıralama
#Saçılım Grafiği
"""
plt.figure()
plt.scatter(range(len(dizili_array)),dizili_array)
plt.grid(True)
plt.ylabel("Yıllar")
plt.title("Olimpiyatlar Çift Yıllarda Düzenlenir")
plt.show()
"""
#Yıl değerlerini datetime veri tipine dönüştürelim        
tarih_saat_nesnesi=pd.to_datetime(veri_zaman["yil"],format ="%Y")
veri_zaman["tarih_saat"]=  tarih_saat_nesnesi     
        
#veri_zaman değişkeninin ana indeksini, datetime tipi olan tarih_saat
# değerine güncelleyelim       
veri_zaman = veri_zaman.set_index("tarih_saat")
veri_zaman.drop(["yil"],axis=1,inplace=True)


#Yıllara Göre Ortalma Yaş, Boy ve Ağırlık Değişimi
""" 
periyodik_veri = veri_zaman.resample("2A").mean()   
   #2 yıllık periyotlar halinde ortalama değerleri alalım 
   
#print(periyodik_veri.head(5))
#                   yas         boy       kilo
# tarih_saat                                  
# 1896-12-31  23.905734  174.280350  72.734056
# 1898-12-31        NaN         NaN        NaN
# 1900-12-31  27.786689  177.882301  74.979950
# 1902-12-31        NaN         NaN        NaN  

#kayıp verileri çıkartalım
periyodik_veri.dropna(axis=0,inplace=True,)    

plt.figure()
periyodik_veri.plot()
plt.title("Yıllara Göre Ortalma Yaş, Boy ve Ağırlık Değişimi")    
plt.xlabel("Yıl")
plt.grid(True)
plt.show()
"""    
  

#Yıllara Göre Madalya Sayıları

veri_zaman = pd.get_dummies(veri_zaman, columns=["madalya"])

periyodik_veri=veri_zaman.resample("2A").sum()
periyodik_veri=periyodik_veri[~(periyodik_veri==0).any(axis=1)]
   #Burda bazı yıllarda değerler 0 olduğu için (periyodik_veri==0) kullanarak
   # o değerleri NaN yaptık akabinde any(axis=1) kullanarak o değerli yılları
   # attık.

print(periyodik_veri.tail()) #Son verileri gösterir

#Yaz - Kış karışık
plt.figure()
periyodik_veri.loc[:,["madalya_Bronze",
                      "madalya_Gold",
                      "madalya_Silver"]].plot()

plt.title("Yıllara Göre Madalya Sayıları")    
plt.xlabel("Yıl")
plt.ylabel("Sayı")
plt.show()

yaz = veri_zaman[veri_zaman.sezon == "Summer"]
kis = veri_zaman[veri_zaman.sezon == "Winter"]

periyodik_veri_kis=kis.resample("A").sum()
periyodik_veri_kis=periyodik_veri_kis[~(periyodik_veri_kis==0).any(axis=1)]

periyodik_veri_yaz=yaz.resample("A").sum()
periyodik_veri_yaz=periyodik_veri_yaz[~(periyodik_veri_yaz==0).any(axis=1)]

#Yaz
plt.figure()
periyodik_veri_yaz.loc[:,["madalya_Bronze",
                      "madalya_Gold",
                      "madalya_Silver"]].plot()

plt.title("Yıllara Göre Madalya Sayıları - Yaz Sezonu")    
plt.xlabel("Yıl")
plt.ylabel("Sayı")
plt.show()  
    
#Kış
plt.figure()
periyodik_veri_kis.loc[:,["madalya_Bronze",
                      "madalya_Gold",
                      "madalya_Silver"]].plot()

plt.title("Yıllara Göre Madalya Sayıları - Kış Sezonu")    
plt.xlabel("Yıl")
plt.ylabel("Sayı")
plt.show() 




    
    