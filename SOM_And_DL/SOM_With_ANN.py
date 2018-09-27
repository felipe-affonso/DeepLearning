#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:28:27 2018

@author: FelipeAffonso
"""

#parte 1 - Identificando fraudes usando SOM

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
frauds = mappings[(1,4)] # esse valor sera baseado na fraude encontrada, cada vez ser uma posição diferente
frauds = sc.inverse_transform(frauds)

# PARTE 2
# UTILIZANDO AS FRAUDES PARA TREINAR UMA REDE NEURAL

# criando as caracteristicas (x)
costumers = dataset.iloc[:, 1:].values

#criando a variavel dependente (y)
is_fraud = np.zeros(len(dataset)) # y
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
    
# Agora podemos treinar a rede neural

# Escalando as variaveis
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
costumers = sc.fit_transform(costumers)

# CRIANDO A ANN
# importando as bibliotecas
import keras
from keras.models import Sequential
from keras.layers import Dense

#Iniciando a ANN
classifier = Sequential()

# Adicionando a input layer e a primeira hidden layer
classifier.add(Dense(units = 2, input_dim = 15, kernel_initializer = 'uniform', activation = 'relu'))

#adicionando a camada de saída
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compilando a ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy',  metrics = ['accuracy'])

# Realizando o fit
classifier.fit(costumers, is_fraud, batch_size = 1, epochs=2)

# PREVENDO RESULTADOS
y_pred = classifier.predict(costumers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)

# ordenando as probabilidades
y_pred = y_pred[y_pred[:, 1].argsort()]

#com isso o setor pode ver os clientes que tem a maior probabilidade de fraude



        
        


    
    
    


