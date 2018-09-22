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

# adicionando a primeira camada da LSTM e o Dropout
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#adicionando uma segunda camada LSTM e o dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#adicionando uma terceira camada LSTM e o dropout
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#adicionando uma quarta camada LSTM e o dropout
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adicionando a camada de saída
regressor.add(Dense(units = 1))

#Compilando a RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fazendo o Fit da RNN nos dados de treino
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32) 

"""
TREINO REALIZADO:
    

"""


# PARTE 3
# PREDIÇÕES

# Pegando os dados de teste
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values #pegando a coluna 'Open'

#Pegando os valores preditos pela rede
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test)-60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60,80): #pegando a partir do dia 60
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualizando os resultados
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


#Treinar depois com o metodo de scoring: scoring = 'neg_mean_squared_error'
