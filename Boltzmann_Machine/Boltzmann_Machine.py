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


# Criando a arquitetura da rede neural

class RBM():
    def __init__(self, nv, nh): #nv = nós visiveis, nh = hidden nodes
        self.W = torch.randn(nh, nv) #pesos aleatorios
        self.a = torch.randn(1, nh) #biad pros nós escondidos
        self.b = torch.randn(1, nv) # bias pros nós visiveis
        
    def sample_h(self, x): # grupos de nós escondidos
        wx = torch.mm(x, self.W.t()) #produto de dois vetores torch
        activation = wx + self.a.expand_as(wx)  #sigmoid
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y): #grupos de nós visiveis
        wy = torch.mm(y, self.W) #produto de dois vetores torch
        activation = wy + self.b.expand_as(wy)  #sigmoid
        p_h_given_h = torch.sigmoid(activation)
        return p_h_given_h, torch.bernoulli(p_h_given_h)
    """
    v0 = vetor contendo as notas de todos os filmes de um usuario
    vk = nós visiveis obtidos após k interações
    ph0 = vetor de probabilidades, que na primeira interação irao valer 1
    phk = probabilidades dos nós escondidos após k interações
    """
    def train(self, v0, vk, ph0, phk): # contrastive divergence
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0-vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
        

#criando o objeto RBM
nv = len(training_set[0]) #nós de entrada
nh = 100 #numero de nós escondidos
batch_size = 100 #tamanho do batch (após quantas interações os pesos serao atualizados)

rbm = RBM(nv, nh)

#treinando o RBM
nb_epoch = 10
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range (0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size] #vetor de notas que os vk usuarios ja deram uma nota
        ph0,_ = rbm.sample_h(v0) #probabilidades iniciais
        
        #for para os k passos da contrastive divergence
        for k in range(10):
             _,hk = rbm.sample_h(vk) #pega os visibles nodes e calcula os hidden nodes
             _,vk = rbm.sample_v(hk) #atualiza os visibles nodes com base nos hidden nodes
             vk[v0 < 0] = v0[v0<0]
         
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        
        # atualizando o erro
        train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
        
#testando a RBM
    



        
