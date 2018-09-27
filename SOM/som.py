#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 21:06:23 2018

@author: FelipeAffonso
"""

#importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importando o dataset
#dataset download from UCI
# Statlog (Australian Credit Approval) Data Set
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
 
#escalando as variaveis
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

#treinando o SOM
# utilizaremos o MiniSom
from minisom import MiniSom
som = MiniSom(x= 10, y = 10, input_len = 15, sigma=1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

#visualizando os resultados
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5, # colocando o marcador no centro do quadrado
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
    
#Encontrando as fraudes
mappings = som.win_map(X)
frauds = mappings[(1,2)]
frauds = sc.inverse_transform(frauds)

    
    
    


