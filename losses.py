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
            mask_array = tf.where(Y_pred[:,:,2] == True, 0 , 1 )
            
            #apply mask to both channels
            for i in range(2):
                Y_pred_masked = tf.math.multiply(Y_pred[:,:,i], tf.cast(mask_array, tf.float32))
                Y_true_masked = tf.math.multiply(Y_pred[:,:,i], tf.cast(mask_array, tf.float32) )
            
            #compute mean 'chi^2'
        
            return (K.mean(K.square(Y_pred_masked-  Y_true_masked  )  ))

        return masked_loss








