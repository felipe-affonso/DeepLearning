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
                        steps_per_epoch=(8000/32), #numero de batches a ser considerados, como utiliamos 32 batches, dividimos os valores por 32
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=(2000/32))

# RODANDO A PRIMEIRA VEZ, SEM NENHUMA ALTERAÇÃO NOS PARAMETROS
"""
Using TensorFlow backend.
Found 8000 images belonging to 2 classes.
Found 2000 images belonging to 2 classes.
Epoch 1/25
250/250 [==============================] - 471s 2s/step - loss: 0.6712 - acc: 0.5869 - val_loss: 0.6196 - val_acc: 0.6605
Epoch 2/25
250/250 [==============================] - 444s 2s/step - loss: 0.5812 - acc: 0.6929 - val_loss: 0.5716 - val_acc: 0.7155
Epoch 3/25
250/250 [==============================] - 432s 2s/step - loss: 0.5523 - acc: 0.7179 - val_loss: 0.5316 - val_acc: 0.7500
Epoch 4/25
250/250 [==============================] - 426s 2s/step - loss: 0.5154 - acc: 0.7441 - val_loss: 0.5033 - val_acc: 0.7550
Epoch 5/25
250/250 [==============================] - 421s 2s/step - loss: 0.4931 - acc: 0.7584 - val_loss: 0.4974 - val_acc: 0.7755
Epoch 6/25
250/250 [==============================] - 418s 2s/step - loss: 0.4835 - acc: 0.7664 - val_loss: 0.4923 - val_acc: 0.7670
Epoch 7/25
250/250 [==============================] - 417s 2s/step - loss: 0.4660 - acc: 0.7749 - val_loss: 0.4642 - val_acc: 0.7810
Epoch 8/25
250/250 [==============================] - 411s 2s/step - loss: 0.4543 - acc: 0.7777 - val_loss: 0.5017 - val_acc: 0.7605
Epoch 9/25
250/250 [==============================] - 445s 2s/step - loss: 0.4366 - acc: 0.7965 - val_loss: 0.4605 - val_acc: 0.7885
Epoch 10/25
250/250 [==============================] - 477s 2s/step - loss: 0.4240 - acc: 0.8027 - val_loss: 0.5102 - val_acc: 0.7690
Epoch 11/25
250/250 [==============================] - 500s 2s/step - loss: 0.4156 - acc: 0.8046 - val_loss: 0.4372 - val_acc: 0.8020
Epoch 12/25
250/250 [==============================] - 493s 2s/step - loss: 0.4108 - acc: 0.8067 - val_loss: 0.4925 - val_acc: 0.7785
Epoch 13/25
250/250 [==============================] - 484s 2s/step - loss: 0.3980 - acc: 0.8200 - val_loss: 0.4541 - val_acc: 0.8015
Epoch 14/25
250/250 [==============================] - 476s 2s/step - loss: 0.3845 - acc: 0.8239 - val_loss: 0.4721 - val_acc: 0.7965
Epoch 15/25
250/250 [==============================] - 476s 2s/step - loss: 0.3718 - acc: 0.8330 - val_loss: 0.4430 - val_acc: 0.8095
Epoch 16/25
250/250 [==============================] - 475s 2s/step - loss: 0.3657 - acc: 0.8340 - val_loss: 0.5130 - val_acc: 0.7795
Epoch 17/25
250/250 [==============================] - 464s 2s/step - loss: 0.3525 - acc: 0.8436 - val_loss: 0.4602 - val_acc: 0.7980
Epoch 18/25
250/250 [==============================] - 423s 2s/step - loss: 0.3542 - acc: 0.8450 - val_loss: 0.4390 - val_acc: 0.8125
Epoch 19/25
250/250 [==============================] - 424s 2s/step - loss: 0.3513 - acc: 0.8421 - val_loss: 0.4437 - val_acc: 0.8115
Epoch 20/25
250/250 [==============================] - 489s 2s/step - loss: 0.3316 - acc: 0.8514 - val_loss: 0.5439 - val_acc: 0.7775
Epoch 21/25
250/250 [==============================] - 478s 2s/step - loss: 0.3159 - acc: 0.8634 - val_loss: 0.4712 - val_acc: 0.8055
Epoch 22/25
250/250 [==============================] - 477s 2s/step - loss: 0.3205 - acc: 0.8571 - val_loss: 0.4891 - val_acc: 0.8095
Epoch 23/25
250/250 [==============================] - 483s 2s/step - loss: 0.3107 - acc: 0.8606 - val_loss: 0.5014 - val_acc: 0.7905
Epoch 24/25
250/250 [==============================] - 453s 2s/step - loss: 0.3017 - acc: 0.8668 - val_loss: 0.4984 - val_acc: 0.7880
Epoch 25/25
250/250 [==============================] - 445s 2s/step - loss: 0.2969 - acc: 0.8740 - val_loss: 0.5966 - val_acc: 0.7735
"""