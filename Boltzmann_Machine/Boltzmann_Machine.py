#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 17:51:13 2018

@author: FelipeAffonso
"""

#importando as bibliotecas

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#importando o dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header= None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header= None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header= None, engine = 'python', encoding = 'latin-1')

# preparando os dados de treino e teste

training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int') #transformando em array

test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(training_set, dtype = 'int') #transformando em array

#obtendo o numero de usuarios e filmes
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

#convertendo os dados em um array com os usuarios em linhas e os filmes em colunas
def convert(data):
    new_data = []
    for id_users in range(1, nb_users+1):
        id_movies = data[:,1][data[:,0] == id_users] #pegando todos os filmes de cada usuario
        id_ratings = data[:,2][data[:,0] == id_users] #pegando todas as notas de cada usuario
        ratings = np.zeros(nb_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

#convertendo os dados em torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
#ao usar o spyder, as duas variaveis irao sumir pois elas ainda não são reconhecidas pela IDE

#convertendo as notas (ratings) para 0 ou 1
training_set[training_set == 0] = -1 #substituindo todos os 0 por -1
training_set[training_set == 1] = 0        
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1 #substituindo todos os 0 por -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


