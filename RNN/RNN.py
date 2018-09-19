#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 18:56:08 2018

@author: FelipeAffonso
"""

#PARTE 1
#importando bibliotecas

import numpy as pd
import matplotlib.pyplot as plt
import pandas as pd

# importando training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values #pegando a coluna 'Open'

