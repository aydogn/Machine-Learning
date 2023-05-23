# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:52:55 2023

@author: aydogn
"""
#Kütüphaneler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import warnings 
warnings.filterwarnings("ignore")

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=(True))#internete erişim için 
import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff


#Veri
veri=pd.read_csv("cwurData.csv")
#Dataframe
df=veri.iloc[:100,:]


#plotly stilleri
pio.templates.default="simple_white"


cizgi1=go.Scatter(x = df.world_rank,
                  y = df.citations,
                  mode = "lines+markers",
                  name = "Alıntı",
                  marker = dict(color = 'rgba(78, 78, 250, 0.85)'),
                  text = df.institution)

cizgi2=go.Scatter(x = df.world_rank,
                  y = df.quality_of_education,
                  mode = "lines+markers",
                  name = "Eğitim-Öğretim Faaliyetleri",
                  marker = dict(color = 'rgba(202, 43, 15, 0.85)'),
                  text = df.institution)

veri=[cizgi1,cizgi2]
yerlesim=dict(title='İlk 100 Üniversitenin Atıf ve Eğitim-Öğretim Puanları', 
              xaxis=dict(title='Dünya Sıralaması', ticklen=5, zeroline=False))
fig=dict(data=veri, layout=yerlesim)

plot(fig, filename='1_çizgi alıntı ve öğretim puanları.html')
















