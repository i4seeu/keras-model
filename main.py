#Load the data
#importing the modules

import numpy as np 
import os
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import adam
from keras.utils import np_utils

#load the data

np.random.seed(100) # for reproducibility
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

''' cifar- 10 has images of airplane , automobile, bird, cat, deer, dog, frog, horse, ship, and truck(10 unique labes)
for each image width = 32 , height = 31 , Number of channels (RGB) = 3 '''

#preprocess data
#flatten the data, MLP doesn't use the 2D structure of the data . 3072 = 3* 32*32

X_train = X_train.reshape(50000,3072) #50,000 images for training
X_test = X_test.reshape(10000,3072) #10, 000 images for testing

#Gaussian Normalization (Z- score)
X_train = (X_train - np.mean(X_train))/np.std(X_train)
X_test = (X_test - np.mean(X_test))/np.std(X_test)

#convert class vectors to binary class matrices (i.e one hot vectors)
labels = 10
Y_train = np_utils.to_categorical(y_train,labels)
Y_test = np_utils.to_categorical(y_test,labels)

#define the model

model = Sequential()
model.add(Dense(512, input_shape=(3072,))) # 3*32*32 = 3072
model.add(Activation('relu'))
model.add(Dropout(0.4)) #regularization
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dropout(0.2)) #regularization
model.add(Dense(labels)) #Last layer with 10 outputs , each output per class
model.add(Activation('sigmoid'))

#compile the model
adam = adam(0.01) #optimizer with a learning rate
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])