import numpy as np
import create_dataset
from create_dataset import Dataset
import CNN_model as CNN
import waterfall_plot
import datetime
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


#make dataset with random masks
Simulated_Data = Dataset(ndata = 200)
X_masked, Y_not_masked = Simulated_Data.generate()

#reshape data and shuffle them
X_masked = np.array(X_masked).reshape(-1, X_masked.shape[1],X_masked.shape[2], 2)
Y_not_masked = np.array(Y_not_masked).reshape(-1, Y_not_masked.shape[1],Y_not_masked.shape[2], 2)


print(X_masked.shape, Y_not_masked.shape)

#create a model with the CNN
CNN = CNN.DeepCNN(shape = X_masked[1:].shape)

CNN.model.summary()
CNN.model.fit(X_masked,Y_not_masked, batch_size = 10, epochs = 1, validation_split = 0.1)




