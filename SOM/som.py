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

