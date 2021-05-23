import numpy as np
import create_dataset
from create_dataset import Dataset
from callbacks import CustomCallbacks
import CNN_model as CNN
import waterfall_plot
import datetime
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from losses import custom_loss



#make dataset with random masks
Simulated_Data = Dataset(ndata = 20)
X_masked, Y_not_masked = Simulated_Data.generate()

#reshape data and shuffle them
X_masked = np.array(X_masked).reshape(-1, X_masked.shape[1],X_masked.shape[2], 2)
Y_not_masked = np.array(Y_not_masked).reshape(-1, Y_not_masked.shape[1],Y_not_masked.shape[2], 2)

print(X_masked.shape, Y_not_masked.shape)

#create an instance of the loss class
loss = custom_loss()

#create a model with the CNN
CNN = CNN.DeepCNN(X_maske.shape[1:], loss)

#note that callbacks can go into the fit step or the model step
custom_callbacks = CustomCallbacks()
modelcheckpoint   = ModelCheckpoint(save_best_only = True, verbose = 1, filepath = '/home/grx40/scratch/HERA_ML/ML_files_testbed/Checkpoints', monitor = 'val_loss')
csvlogger = CSVLogger(separator = ',', filename = 'run/log.csv')
callback_list  = [custom_callbacks, modelcheckpoint , csvlogger]


CNN.model.summary()
CNN.model.fit(X_masked,Y_not_masked, batch_size = 10, epochs = 1, callbacks = [callback_list], validation_split = 0.1)




