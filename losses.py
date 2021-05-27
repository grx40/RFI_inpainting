import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf

class custom_loss:

    def wrapper(self):

        def masked_loss(Y_pred, Y_true):
            '''
                This loss function takes the prediction Y_pred, applies the mask to the first two channels
                which correspond to the real and imag part of the visibilities, then computes the 'chi^2'
                of those two channels.
            '''
            #mask_array = tf.where(Y_pred[:,:,:,2] == 1, 0 , 1 )
            mask_array = Y_true[:,:,:,0:2]
            
            #apply mask to the amplitude and visibility channels
            Y_pred_masked = tf.math.multiply(Y_pred[:,:,:,0:2], tf.cast(mask_array, tf.float32))
            Y_true_masked = tf.math.multiply(Y_true[:,:,:,0:2], tf.cast(mask_array, tf.float32) )
            
            #compute mean 'chi^2'
            print('After loop' ,Y_pred_masked ,Y_true_masked )
        
            return K.mean(K.square(K.abs((Y_pred_masked - Y_true_masked))))

        return masked_loss








