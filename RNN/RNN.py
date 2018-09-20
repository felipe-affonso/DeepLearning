#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 18:56:08 2018

@author: FelipeAffonso
"""

#PARTE 1
#importando bibliotecas

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importando training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values #pegando a coluna 'Open'

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))
training_set_scaled = sc.fit_transform(training_set)

#criando timesteps 
X_train = []
y_train = []
for i in range(60,1258): #pegando a partir do dia 60
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)
    
#reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    

#PARTE 2
#importando as bibliotecas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#inicializando a RNN
regressor = Sequential()

