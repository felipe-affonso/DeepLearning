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

#PARTE 4
# AVALIANDO A ANN

#importando as bibliotecas
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss='binary_crossentropy',  metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size = 10, epochs=100)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()

#0.8441249952092766
#0.016974705113547123


# Dropout

from keras.layers import Dropout

#Iniciando a ANN
classifier = Sequential()

# Adicionando a input layer e a primeira hidden layer
classifier.add(Dense(units = 6, input_dim = 11, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

#adicionando uma segunda camada "escondida"
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

#adicionando a camada de saída
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compilando a ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy',  metrics = ['accuracy'])


# PARTE 5
# TUNNING THE ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss='binary_crossentropy',  metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)

parameters = {'batch_size': [25,32],
              'epochs': [100, 500],
              'optimizer': ['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

#best _accuracy 0.852125
# best param: 
# batch_size 25
# epochs 500
# optimizer adam











    
    
    
    