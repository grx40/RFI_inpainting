import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf

class custom_loss:

    def wrapper(self):

        def masked_loss(Y_true, Y_pred):
            '''
                This loss function takes the prediction Y_pred, applies the mask to the first two channels
                which correspond to the real and imag part of the visibilities, then computes the 'chi^2'
                of those two channels.
            '''
            
            #make an array of ones
            ones = tf.ones_like(Y_true[:,:,:,3])
            #invert the mask, we want only the masked areas to enter the chi^2
            mask_array = ones - Y_true[:,:,:,3]
            
            #K.print_tensor(ones, message='ones = ')
            #K.print_tensor(Y_true[0,:,:,3], message='y_true[0,:,:,3] = ')
            #K.print_tensor(mask_array[0], message='mask_array = ')
            
            
            #apply mask to the amplitude and visibility channels
            #loop this in the future, but for now, whatever
            for i in range(2):
                Y_pred_masked = tf.math.multiply(Y_pred[:,:,:,i], tf.cast(mask_array, tf.float32))
                Y_true_masked = tf.math.multiply(Y_true[:,:,:,i], tf.cast(mask_array, tf.float32) )
            
            #Y_pred_masked = tf.math.multiply(Y_pred[:,:,:,1], tf.cast(mask_array, tf.float32))
            #Y_true_masked = tf.math.multiply(Y_true[:,:,:,1], tf.cast(mask_array, tf.float32) )
            
            #compute mean 'chi^2'
            
            #K.print_tensor(Y_true_masked, message='y_true_masked = ')
            #K.print_tensor(K.mean(K.square(K.abs((Y_pred_masked - Y_true_masked)))), message='loss = ')
            
        
            return K.mean(K.square(K.abs((Y_pred_masked - Y_true_masked))))

        return masked_loss








