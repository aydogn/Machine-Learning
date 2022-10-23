"""Bu dosyada olmayan verileri diğer verilerin ortalamasını alarak 
yerleştirdik. Ancak tabii ki farklı ülkelerin verileri diğer ülkeleri 
etkilemeyebilir. Bu yüzden her ülkenin verisinin ortalamasını alarak 
o ülkenin bilinmeyenine aktardık."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Eksik değerleri, her sütun boyunca açıklayıcı bir istatistik 
 (örneğin, ortalama, medyan veya en sık) kullanarak veya sabit bir 
 değer kullanarak değiştirin."""
 
"""from sklearn.impute import SimpleImputer

veriler = pd.read_csv('eksikveriler.csv')

#missing_values=Eksik değerler için yer tutucu
#strategy =  “mean” “median” “most_frequent” “constant” değerlerini alır
imputer = SimpleImputer(missing_values = np.nan , strategy="mean")

Yas1 = veriler.iloc[9:15,1:4].values
Yas2 = veriler.iloc[15:,1:4].values


#fit() arkaplanda işlem gerçekleştirir ancak göstermez yazmaz işlemez
imputer1 = imputer.fit(Yas1[:,:])

#transform() gerçekleştirilen işlemi göstermeye yarar
Yas1[:,:] = imputer1.transform(Yas1[:,:])
#print(Yas1)

print('-----------------------------------------------')

imputer2 = imputer.fit(Yas2[:,:])
Yas2[:,:] = imputer2.transform(Yas2[:,:])
#print(Yas2)"""

#########################################################################

"""Eğer ki bütün değerlerin ortalamasını alarak eksikleri tamamlamak
istersek alttak yazılım devreye girmeli."""

from sklearn.impute import SimpleImputer
veriler = pd.read_csv('eksikveriler.csv')
all_imputer = SimpleImputer(missing_values = np.nan , strategy="mean")

Yas = veriler.iloc[:,1:4].values

#all_imputer = all_imputer.fit(Yas[:,1:4])

Yas[:,1:4]= all_imputer.fit_transform(Yas[:,1:4])
print(Yas)
