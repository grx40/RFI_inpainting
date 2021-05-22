import numpy as np
import datetime
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from misc import custom_loss_wrapper


class DeepCNN(Sequential):

    #this is not a very extensive list right now but let us only require that the user specify the bare
    #minimum. We'll make this model more flexible in the future
    def __init__(self, shape, X, **kwargs):
        super(DeepCNN, self).__init__()
        #make sure RGB images have 3 channels at the end as input
        self.shape = shape
        self.loss = kwargs.pop('loss', 'binary_crossentropy')
        self.optimizer  = kwargs.pop('optimizer', 'adam')
        self.CNN_metrics = kwargs.pop('metrics', 'accuracy')
        self.mask_array = np.getmask(X)

        #initialize the model from tensorflow. Vanilla sequential
        self.model = Sequential()
        
        #add first layer, convolution layer: 64 layers dense, 3x3 Kernel
        self.model.add(Conv2D(64 , (20,20), padding = 'same', input_shape = self.shape  , name = 'layer1' ))
        self.model.add(Activation('relu'))

        #add second layer
        #why don't I need to initialize these arrays with a bunch of values? (apparently it defaults)
        self.model.add(Conv2D(64, (15,15), padding = 'same', name = 'layer2'))
        self.model.add(Activation('relu'))

        #3rd layer
        #try initializing the arrays within the add method
        self.model.add(Conv2D(64, (15,15), padding = 'same', name = 'layer3'))
        self.model.add(Activation('relu'))

        #4th layer
        self.model.add(Conv2D(64, (10,10), padding = 'same', name = 'layer4'))
        self.model.add(Activation('relu'))

        #5th layer
        self.model.add(Conv2D(64, (6,6), padding = 'same', name = 'layer5'))
        self.model.add(Activation('relu'))

        #6th layer
        self.model.add(Conv2D(64, (4,4), padding = 'same', name = 'layer6'))
        self.model.add(Activation('relu'))

        #7th layer
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))

        #7th layer
        self.model.add(Dense(2))
        
        masked_loss = custom_loss_wrapper(mask_array)
        
        self.model.compile(loss = masked_loss, optimizer = self.optimizer, metrics = [masked_loss])

        self.model.build()
    
                   
    









