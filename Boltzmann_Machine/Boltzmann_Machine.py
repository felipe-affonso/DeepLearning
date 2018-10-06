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

