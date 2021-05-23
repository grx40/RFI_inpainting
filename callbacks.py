import numpy as np
import tensorflow as tf
from tensorflow import keras

class CustomCallbacks(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))
