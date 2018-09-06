#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 10:15:43 2018

@author: FelipeAffonso
"""


# PARTE 1
# importando as bibliotecas e pacotes
from keras.models import Sequential 
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#inicializando a CNN
classifier = Sequential()

# Passo 1 - Convolution
classifier.add(Convolution2D(32,(3,3),input_shape = (64, 64, 3), activation = 'relu'))

# Passo 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# adicionando uma segunda Convolution Layer
classifier.add(Convolution2D(32,(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Passo 3 - Flattening
classifier.add(Flatten())

# Passo 4 - ANN
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Passo 5 - Compilando a rede 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Parte 2
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000)

#acc: 0.8516
#val_acc = 0.8180

