#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:49:19 2018

@author: FelipeAffonso
"""

# PARTE 1 - PRE PROCESSAMENTO DOS DADOS
# Importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importando o Dataset
dataset = pd.read_csv('Bank_Dataset.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Codificando categorias
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#evitando a dummy variable trap
X = X[:, 1:]

# Dividindo o dataset em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalando as variaveis
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# PARTE 2
# CRIANDO A ANN
# importando as bibliotecas
import keras
from keras.models import Sequential
from keras.layers import Dense

#Iniciando a ANN
classifier = Sequential()

# Adicionando a input layer e a primeira hidden layer
classifier.add(Dense(units = 6, input_dim = 11, kernel_initializer = 'uniform', activation = 'relu'))

#adicionando uma segunda camada "escondida"
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

#adicionando a camada de saída
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compilando a ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy',  metrics = ['accuracy'])

# Realizando o fit
classifier.fit(X_train, y_train, batch_size = 10, epochs=100)

#RESULTADO:
#Epoch 100/100
#8000/8000 [==============================] - 2s 291us/step - loss: 0.4008 - acc: 0.8345
#Out[14]: <keras.callbacks.History at 0x104eeeb70>

# PARTE 3
# PREVENDO RESULTADOS
y_pred = classifier.predict(X_test)

#converter as probabilidades para 0 ou 1
y_pred = (y_pred>0.5) 

# criando a matriz de confusão
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#testando o modelo nos dados de teste
#calculo da accuracy (1531+154)/2000
# resultado: 0.8425
# melhor do que o modelo dos dados de treino.

#REALIZANDO UMA SIMULAÇÃO
simulation = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
simulation = (simulation>0.5)