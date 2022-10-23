
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('veriler.csv')

"""boy = veriler[['boy']]
boykilo = veriler[['boy','kilo']]
print(boykilo)"""

class insan:
    boy=174
    def kos(self,v):
        return v+10
    
ali = insan()

print(ali.boy)
print(ali.kos(10))