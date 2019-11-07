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

